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

# Sample questions configuration
SAMPLE_QUESTIONS = [
    ("pumpkin", "Chef, how do I make your pumpkin and miso soup?", "🎃 Pumpkin Soup"),
    ("scrambled", "How do I make the Scrambled Eggs with Miso Butter?", "🍳 Miso Butter Eggs"),
    ("paneer", "Chef, what is the recipe for the Paneer Miso Dip?", "🧀 Paneer Miso Dip"),
    ("noodles", "How do I elevate my instant noodles using toasted sesame miso?", "🍜 Elevate Noodles"),
    ("prawns", "How should I prepare sweet water prawns before cooking?", "🍤 Prepare Prawns"),
    ("chocolate", "Can you share the recipe for the Chocolate Miso Butter?", "🍫 Chocolate Miso Butter"),
    ("devilled", "How do I make your Miso Devilled Eggs?", "🥚 Miso Devilled Eggs")
]

@cl.on_chat_start
async def start():
    # Set default session values
    cl.user_session.set("collection_name", "recipes")
    cl.user_session.set("top_k", 5)
    cl.user_session.set("temperature", 0.3)
    cl.user_session.set("chat_history", [])

    # Send settings panel configurations
    await cl.ChatSettings([
        Select(
            id="collection_name",
            label="Query Collection",
            values=["recipes", "instagram_reels"],
            initial_index=0,
            description="'recipes' has recipe details. 'instagram_reels' contains Reels transcriptions."
        ),
        Slider(
            id="top_k",
            label="PostgreSQL Retrievals (Top K)",
            initial=5,
            min=2,
            max=8,
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
        rag_chat.load_collection(collection_name="recipes")
    except SystemExit:
        await cl.ErrorMessage(
            content="PostgreSQL initialization failed. Run `python classify_reels.py` in your terminal to ingest data."
        ).send()
        return

    # Build and send welcome message with action buttons
    actions = [
        cl.Action(name="ask_suggested", payload={"value": q[1]}, label=q[2])
        for q in SAMPLE_QUESTIONS
    ]
    
    welcome_content = (
        "🍳 **Welcome to Ground Up — Talk to the Chef!**\n\n"
        "I'm the head chef and founder of Ground Up. Ask me anything about our recipes, "
        "ingredients, or cooking tips! Use the settings panel on the bottom-left to adjust parameters.\n\n"
        "Select one of these questions to get started:"
    )
    
    await cl.Message(content=welcome_content, actions=actions).send()


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("collection_name", settings["collection_name"])
    cl.user_session.set("top_k", int(settings["top_k"]))
    cl.user_session.set("temperature", float(settings["temperature"]))
    
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
    collection_name = cl.user_session.get("collection_name") or "recipes"
    top_k = cl.user_session.get("top_k") or 5
    temperature = cl.user_session.get("temperature") or 0.3
    
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
        chunks = rag_chat.retrieve(collection, query_text, k=top_k)
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
            
        # Call LLM via unified ask_gemini
        try:
            answer_text = rag_chat.ask_gemini(query_text, context, llm_history, temperature=temperature)
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

        for idx, source in enumerate(unique_sources):
            r_id = source["reel_id"]
            mp4_path = BASE_DIR / "downloads" / f"{r_id}.mp4"
            jpg_path = BASE_DIR / "downloads" / f"{r_id}.jpg"
            
            # Use appropriate media element
            if mp4_path.exists():
                elements.append(cl.Video(name=f"Video: {r_id}", path=str(mp4_path), display="inline"))
            elif jpg_path.exists():
                elements.append(cl.Image(name=f"Thumbnail: {r_id}", path=str(jpg_path), display="inline"))
                
            # Formatting source metadata text block for the sidebar drawer
            recipe_name = source.get("recipe_name")
            recipe_line = f"🍳 **Recipe:** {recipe_name}\n" if recipe_name else ""
            
            source_desc = (
                f"{recipe_line}"
                f"📅 **Date:** {source.get('date', 'Unknown')}\n"
                f"❤️ **Likes:** {source.get('likes', 'N/A')}\n"
                f"📈 **Relevance Score:** {source['score']:.3f}\n"
                f"🔗 [Instagram Link]({source['url']})\n\n"
                f"**Context Chunk:**\n{source['text']}"
            )
            elements.append(cl.Text(name=f"Source {idx+1}: {r_id}", content=source_desc, display="side"))

    # Send final answer with attachments
    await cl.Message(content=answer_text, elements=elements).send()
