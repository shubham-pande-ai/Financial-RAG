import streamlit as st
import requests

SERVER_URL = "http://localhost:8001"

# --- Page Config ---
st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial RAG Assistant")
st.caption("Ask questions about company financial reports using advanced Retrieval-Augmented Generation.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper to fetch available providers ---
@st.cache_data(ttl=300)
def get_providers():
    try:
        r = requests.get(f"{SERVER_URL}/providers", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    # Fallback if server is down or endpoint fails
    return [
        {"id": "groq-llama", "label": "Groq — llama-3.3-70b-versatile ★ FASTEST"},
        {"id": "or-llama70b", "label": "OpenRouter — Llama 3.3 70B [FREE] ★ BEST"},
        {"id": "or-gemini", "label": "OpenRouter — Gemini 2.0 Flash [FREE]"},
        {"id": "gemini", "label": "Google Gemini — gemini-2.0-flash (direct)"},
    ]

providers = get_providers()
provider_options = {p["label"]: p["id"] for p in providers}

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Query Settings")
    
    selected_provider_label = st.selectbox(
        "LLM Provider",
        options=list(provider_options.keys()),
        index=0,
        help="Select the AI model used to generate the final answer."
    )
    
    symbol = st.text_input("Company Symbol (e.g., ADANIPORTS)", value="", help="Leave blank to search across all indexed companies.")
    
    doc_type = st.selectbox(
        "Document Type",
        options=["both", "annual_report", "concall"],
        index=0
    )
    
    year = st.text_input("Year (Optional)", value="")
    
    st.divider()
    st.markdown("""
    **Architecture:**
    1. **Retrieval**: Searches Qdrant Vector DB
    2. **Reranking**: Uses local cross-encoder
    3. **Generation**: Uses selected LLM
    """)

# --- Helper to format source dictionaries ---
def format_source(idx, source):
    if isinstance(source, dict):
        sym = source.get('symbol', 'UNKNOWN')
        yr = source.get('year', 'N/A')
        doc = source.get('doc_type', 'Document').replace('_', ' ').title()
        sec = source.get('section', 'N/A')
        pg = source.get('page', 'N/A')
        score = source.get('score', 0)
        return f"**[{idx+1}] {sym} ({yr})** — *{doc} (Page {pg})*  \n`Section: {sec}` | `Score: {score:.1%}`"
    return f"**[{idx+1}]** {source}"


# --- Chat Interface ---
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if available
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📄 View extracted sources ({msg['chunks_used']} chunks)"):
                for idx, source in enumerate(msg["sources"]):
                    st.markdown(format_source(idx, source))
        
        # Display meta info if available
        if "meta" in msg:
            st.caption(f"⏱️ {msg['meta']['latency_sec']}s | 🧠 {msg['meta']['model_used']}")

# --- Handle User Input ---
if prompt := st.chat_input("E.g., What was the revenue for FY25?"):
    
    # 1. Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call backend and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Check if server is up
        try:
            requests.get(f"{SERVER_URL}/health", timeout=2)
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend server at {SERVER_URL}. Make sure `python server.py` is running!")
            st.stop()
            
        with st.spinner("Retrieving documents and generating answer..."):
            payload = {
                "query": prompt,
                "provider": provider_options[selected_provider_label],
                "doc_type": doc_type
            }
            
            # Only add if provided
            if symbol.strip():
                payload["symbol"] = symbol.strip()
            if year.strip() and year.isdigit():
                payload["year"] = int(year.strip())
                
            try:
                # The backend might take 10-20 seconds depending on LLM
                resp = requests.post(f"{SERVER_URL}/query", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                
                answer = data.get("answer", "No answer provided.")
                sources = data.get("sources", [])
                chunks_used = data.get("chunks_used", 0)
                
                # Display the main answer
                message_placeholder.markdown(answer)
                
                # Show sources in expander (proving extraction)
                if sources:
                    with st.expander(f"📄 View extracted sources ({chunks_used} chunks)"):
                        for idx, source in enumerate(sources):
                            st.markdown(format_source(idx, source))
                            
                # Show metadata
                meta_str = f"⏱️ Latency: {data.get('latency_sec', 'N/A')}s | 🧠 Model: {data.get('model_used', 'N/A')}"
                st.caption(meta_str)
                
                # Save to session state so it persists
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources,
                    "chunks_used": chunks_used,
                    "meta": {
                        "latency_sec": data.get("latency_sec"),
                        "model_used": data.get("model_used")
                    }
                })
                
            except requests.exceptions.Timeout:
                st.error("Request timed out. The LLM provider might be slow or down.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
