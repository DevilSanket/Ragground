import os
import re
import sys
from pathlib import Path
import streamlit as st

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
try:
    import rag_chat
except ImportError:
    st.error("Could not import rag_chat.py. Please ensure you are running from the pipeline directory.")
    sys.exit(1)

# ─── Page Settings & Aesthetics ────────────────────────────────────────────────
st.set_page_config(
    page_title="Ground Up Chef RAG Bot",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (amber/orange theme matching Ground Up food brand)
st.markdown("""
<style>
    .reportview-container {
        background: #faf8f5;
    }
    .stApp {
        background-color: #fcfbf9;
    }
    h1, h2, h3 {
        color: #8c2d19;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .chat-bubble {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    .chef-bubble {
        background-color: #fff6f0;
        border-left: 5px solid #d95d39;
    }
    .user-bubble {
        background-color: #f0f2f6;
        border-left: 5px solid #5a6b7c;
    }
    .source-container {
        border: 1px solid #ebdcd5;
        border-radius: 8px;
        padding: 10px;
        background-color: #ffffff;
        margin-bottom: 15px;
    }
    .badge {
        background-color: #d95d39;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── App State Initialisation ──────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ─── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://groundup.in/cdn/shop/files/GroundUp_Logo_Trans.png?v=1686737525", width=150)
    st.title("Chef's Kitchen")
    st.write("Interact with the Ground Up RAG Bot using Instagram Reels, Captions, and Videos.")
    st.markdown("---")
    
    st.subheader("Database Configuration")
    collection_name = st.selectbox(
        "Query Collection",
        options=["recipes", "instagram_reels"],
        index=0,
        help="'recipes' contains the active recipe knowledge base. 'instagram_reels' is disabled in this run."
    )
    
    top_k = st.slider("PostgreSQL Retrievals (Top K)", min_value=2, max_value=8, value=5)
    temperature = st.slider("LLM Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    st.markdown("---")
    st.subheader("Content Filter")
    content_type_val = st.selectbox(
        "Filter by type",
        options=["all", "recipe", "travel_vlog", "informational", "product_showcase", "other"],
        index=0,
        help="Filter responses to a specific content type, or 'all' to search across everything."
    )
    content_type_filter = None if content_type_val == "all" else content_type_val
    
    st.markdown("---")
    st.subheader("Suggested Questions")
    
    # Preset questions list
    sample_questions = [
        "Chef, how do I make your pumpkin and miso soup?",
        "How do I make the Scrambled Eggs with Miso Butter?",
        "Chef, what is the recipe for the Paneer Miso Dip?",
        "How do I elevate my instant noodles using toasted sesame miso?",
        "How should I prepare sweet water prawns before cooking?",
        "Can you share the recipe for the Chocolate Miso Butter?",
        "How do I make your Miso Devilled Eggs?"
    ]
    
    # Preset click handlers
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.last_query = q

# ─── Load Vector DB Collection ───────────────────────────────────────────────
@st.cache_resource
def get_cached_collection(name):
    # Overrides default collection name
    return rag_chat.load_collection(collection_name=name)

try:
    collection = get_cached_collection(collection_name)
except SystemExit:
    st.error("PostgreSQL initialization failed. Have you run the classification stage?")
    st.info("Run in terminal: `python classify_reels.py` or `python run_pipeline.py --stages classify`")
    st.stop()

# ─── Main Interface ────────────────────────────────────────────────────────────
st.title("🍳 Ground Up — Talk to the Chef")
st.write("Welcome to my kitchen! Ask me anything about our recipes, cooking techniques, or Ground Up ingredients.")

# Display Chat History
for turn in st.session_state.chat_history:
    role = turn["role"]
    avatar = "👨‍🍳" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(turn["content"])

# Query Processing Logic
def handle_query(query_text):
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not configured. Please add it to your reels_pipeline/.env file.")
        return

    # Add user message to UI state
    st.session_state.chat_history.append({"role": "user", "content": query_text})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(query_text)

    # Process and retrieve
    with st.spinner("Talking to the chef..."):
        # Retrieve chunks
        chunks = rag_chat.retrieve(collection, query_text, k=top_k, content_type=content_type_filter)
        context = rag_chat.build_context(chunks)
        
        # Build LLM History
        llm_history = []
        # Keep last 4 turns for context memory
        for turn in st.session_state.chat_history[-8:-1]:
            llm_history.append({
                "role": turn["role"],
                "text": turn["content"]
            })
            
        # Call LLM via unified ask_gemini
        try:
            answer_text = rag_chat.ask_gemini(query_text, context, llm_history, temperature=temperature, chunks=chunks)
        except Exception as e:
            answer_text = f"Chef's assistant failed to get a response: {e}"

    # Add assistant response to UI state
    st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
    
    with st.chat_message("assistant", avatar="👨‍🍳"):
        st.markdown(answer_text)

    # ─── Render Media and Sources ─────────────────────────────────────────────
    if chunks:
        st.markdown("---")
        st.subheader("📚 Source Media & Context")
        
        # Deduplicate retrieved reels/posts to avoid playing the same video multiple times
        seen_reels = set()
        unique_sources = []
        for c in chunks:
            r_id = c["reel_id"]
            if r_id and r_id not in seen_reels:
                seen_reels.add(r_id)
                unique_sources.append(c)

        # Create columns for unique sources
        cols = st.columns(len(unique_sources))
        for idx, source in enumerate(unique_sources):
            with cols[idx]:
                r_id = source["reel_id"]
                st.markdown(f"#### Source {idx+1}: `{r_id}`")
                if source.get("recipe_name"):
                    st.caption(f"🍳 **Recipe:** {source['recipe_name']}")
                
                # Check for local video (.mp4)
                mp4_path = BASE_DIR / "downloads" / f"{r_id}.mp4"
                jpg_path = BASE_DIR / "downloads" / f"{r_id}.jpg"
                
                if mp4_path.exists():
                    st.write("🎥 **Watch Video:**")
                    st.video(str(mp4_path))
                elif jpg_path.exists():
                    st.write("🖼️ **Post Thumbnail:**")
                    st.image(str(jpg_path), use_container_width=True)
                else:
                    st.info("No local media file available.")
                
                # Metadata block
                st.markdown(f"""
                * **Date:** {source.get('date', 'Unknown')}
                * **Likes:** {source.get('likes', 'N/A')}
                * **Relevance:** `{source['score']:.3f}`
                * [Instagram Link]({source['url']})
                """)
                
                with st.expander("Show Retrieved Text"):
                    st.caption(source["text"])

# Handle text input
query = st.chat_input("Ask the chef a question...")
if query:
    handle_query(query)
elif st.session_state.last_query:
    # Handle click on sample question
    q = st.session_state.last_query
    st.session_state.last_query = "" # clear state
    handle_query(q)
