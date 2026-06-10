import os
import re
import sys
from pathlib import Path
import chainlit as cl
from chainlit.input_widget import Select, Slider

# ─── Load Environment Configuration ───────────────────────────────────────────
BASE_DIR = Path(__file__).parent
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Add pipeline dir to path to import RAG functions
sys.path.append(str(BASE_DIR))
import rag_chat
import postgres_db

# Sample questions — span all content types
SAMPLE_QUESTIONS = [
    # Recipes
    ("pumpkin",    "Chef, how do I make your pumpkin and miso soup?",             "🎃 Pumpkin Miso Soup"),
    ("scrambled",  "How do I make the Scrambled Eggs with Miso Butter?",           "🍳 Miso Butter Eggs"),
    ("paneer",     "What is the recipe for your Paneer Miso Dip?",                 "🧀 Paneer Miso Dip"),
    ("noodles",    "How do I elevate instant noodles with toasted sesame miso?",   "🍜 Elevated Noodles"),
    # Travel
    ("travel",     "Tell me about any food markets or places you've explored.",    "🌍 Food Markets"),
    # Informational
    ("miso",       "What is miso and how do you ferment it at Ground Up?",         "📚 What is Miso?"),
    ("ferment",    "Why do you believe in fermented foods? What are the benefits?", "🧪 Fermentation"),
    # Product showcase
    ("product",    "Tell me about your latest Ground Up products.",                "🛍️ New Products"),
]

@cl.on_chat_start
async def start():
    # Set default session values
    cl.user_session.set("collection_name", "instagram_reels")
    cl.user_session.set("content_type",    None)   # None = all types
    cl.user_session.set("top_k",           5)
    cl.user_session.set("temperature",     0.3)
    cl.user_session.set("chat_history",    [])

    # Send settings panel configurations
    await cl.ChatSettings([
        Select(
            id="content_type",
            label="Content Filter",
            values=["all", "recipe", "travel_vlog", "informational", "product_showcase", "other"],
            initial_index=0,
            description="Filter responses to a specific content type, or 'all' to search across everything."
        ),
        Select(
            id="collection_name",
            label="Vector DB Collection",
            values=["instagram_reels"],
            initial_index=0,
            description="The vector database collection to search."
        ),
        Slider(
            id="top_k",
            label="Retrievals (Top K)",
            initial=5,
            min=2,
            max=10,
            step=1
        ),
        Slider(
            id="temperature",
            label="LLM Temperature (Creativity)",
            initial=0.3,
            min=0.0,
            max=1.0,
            step=0.1
        )
    ]).send()

    # Pre-verify collection loading
    try:
        rag_chat.load_collection(collection_name="instagram_reels")
    except SystemExit:
        await cl.ErrorMessage(
            content="⚠️ Vector DB not initialised. Run `python classify_reels.py` to ingest data."
        ).send()
        return

    # Build and send welcome message with action buttons
    actions = [
        cl.Action(name="ask_suggested", payload={"value": q[1]}, label=q[2])
        for q in SAMPLE_QUESTIONS
    ]
    
    welcome_content = (
        "🍳 **Welcome to Ground Up — Talk to the Founder!**\n\n"
        "I'm the founder of Ground Up — ask me anything about our **recipes**, "
        "**travel stories**, **ingredient knowledge**, or **products**!\n\n"
        "Use the ⚙️ settings panel to filter by content type or adjust parameters.\n\n"
        "Try one of these questions to get started:"
    )
    
    await cl.Message(content=welcome_content, actions=actions).send()


@cl.on_settings_update
async def setup_agent(settings):
    ct_raw = settings.get("content_type", "all")
    cl.user_session.set("content_type",    None if ct_raw == "all" else ct_raw)
    cl.user_session.set("collection_name", settings["collection_name"])
    cl.user_session.set("top_k",           int(settings["top_k"]))
    cl.user_session.set("temperature",     float(settings["temperature"]))

    # Attempt loading collection
    try:
        rag_chat.load_collection(collection_name=settings["collection_name"])
    except SystemExit:
        await cl.ErrorMessage(content=f"⚠️ Failed to load database collection: {settings['collection_name']}").send()


@cl.action_callback("ask_suggested")
async def on_action(action):
    query_text = action.payload.get("value", "")
    # Remove actions from the welcome message
    await action.remove()
    # Execute query
    await handle_query(query_text)


@cl.on_message
async def main(message: cl.Message):
    await handle_query(message.content)


async def handle_query(query_text: str):
    collection_name = cl.user_session.get("collection_name") or "instagram_reels"
    content_type    = cl.user_session.get("content_type")     # None = all types
    top_k           = cl.user_session.get("top_k") or 5
    temperature     = cl.user_session.get("temperature") or 0.3
    
    if not os.environ.get("GEMINI_API_KEY"):
        await cl.ErrorMessage(content="❌ GEMINI_API_KEY is not configured in your .env file.").send()
        return

    # Create thinking step
    async with cl.Step(name="Thinking", show_input=True) as step:
        step.input = query_text
        
        # Load collection
        try:
            collection = rag_chat.load_collection(collection_name=collection_name)
        except SystemExit:
            step.output = "Error loading database collection."
            await cl.ErrorMessage(content="❌ Database collection failed to load. Please verify PostgreSQL connection.").send()
            return
            
        # Retrieve chunks
        chunks = rag_chat.retrieve(collection, query_text, k=top_k, content_type=content_type)
        context = rag_chat.build_context(chunks)
        
        # Build LLM History (map assistant -> model for Gemini SDK compatibility)
        history = cl.user_session.get("chat_history") or []
        llm_history = []
        for turn in history[-8:]:
            role = "model" if turn["role"] in ("model", "assistant") else "user"
            llm_history.append({
                "role": role,
                "text": turn["content"]
            })
            
        # Call LLM via unified ask_gemini (chunks enable adaptive system prompt)
        try:
            answer_text = rag_chat.ask_gemini(query_text, context, llm_history,
                                              temperature=temperature, chunks=chunks)
        except Exception as e:
            answer_text = f"Chef's assistant failed to get a response: {e}"
            
        step.output = f"Retrieved {len(chunks)} relevant database context chunks."

    # Update session chat history
    history.append({"role": "user", "content": query_text})
    history.append({"role": "assistant", "content": answer_text})
    cl.user_session.set("chat_history", history)

    # Attach sources
    elements = []
    if chunks:
        seen_reels = set()
        unique_sources = []
        for c in chunks:
            r_id = c["reel_id"]
            if r_id and r_id not in seen_reels:
                seen_reels.add(r_id)
                unique_sources.append(c)

        sources_list = []
        for idx, source in enumerate(unique_sources):
            r_id = source["reel_id"]
            mp4_path = BASE_DIR / "downloads" / f"{r_id}.mp4"
            jpg_path = BASE_DIR / "downloads" / f"{r_id}.jpg"
            
            # Use appropriate media element
            if mp4_path.exists():
                elements.append(cl.Video(name=f"Video: {r_id}", path=str(mp4_path), display="inline"))
            elif jpg_path.exists():
                elements.append(cl.Image(name=f"Thumbnail: {r_id}", path=str(jpg_path), display="inline"))
                
            # Build clean metadata representation for the inline text footer
            title = (
                source.get("recipe_name") or 
                source.get("location") or 
                source.get("subject") or 
                source.get("product_name") or 
                f"Reel {r_id}"
            )
            url = source.get("url", "")
            score = source.get("score", 0.0)
            date_str = f" ({source['date']})" if source.get("date") else ""
            
            if url:
                sources_list.append(f"- [{title}]({url}){date_str} (Score: {score:.3f})")
            else:
                sources_list.append(f"- {title}{date_str} (Score: {score:.3f})")

        if sources_list:
            sources_md = "\n\n---\n**Sources:**\n" + "\n".join(sources_list)
            answer_text += sources_md

    # Send final answer with attachments
    await cl.Message(content=answer_text, elements=elements).send()
