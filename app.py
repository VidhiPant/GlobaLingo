import streamlit as st
import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuration & API Setup
API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = API_KEY
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
PERSIST_DIR = "./nexus_vector_db"

st.set_page_config(page_title="GlobaLingo", page_icon="🌐", layout="wide")

# 2. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# 3. Sidebar Navigation & Multi-mode logic
with st.sidebar:
    st.title("🌐 GlobaLingo")
    app_mode = st.radio("Navigation", ["🏠 Home", "💬 Chatbot"])
    st.divider()
    
    # Multilingual Selection (Only active in Chatbot mode)
    LANGUAGES = ["English", "German", "French", "Spanish", "Hindi", "Japanese", "Chinese", "Arabic"]
    target_lang = st.selectbox("Search Output Language:", sorted(LANGUAGES))
    
    # File Uploader for RAG
    st.subheader("📚 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save temp file for Loader
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            
            # Create/Update Vector Store
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=OpenAIEmbeddings(),
                persist_directory=PERSIST_DIR
            )
            st.session_state.vector_db = vectorstore
            st.success("File processed! You can now ask questions about it.")

# --- MODE: HOME (English Only) ---
if app_mode == "🏠 Home":
    st.title("🏠 Welcome to Globalingo")
    st.markdown("""
    I am your central hub for intelligence. Currently, we are in **Home Mode** (English).
    
    **How to use:**
    1. **Upload a Document** in the sidebar to add it to my memory.
    2. **Switch to Chatbot** in the sidebar to chat in different languages.
    3. Ask me to **summarize** or **translate** specific parts of your files!
    """)
    st.info("I will communicate in English here. Switch modes for multilingual support.")

# --- MODE: CHATBOT (Multilingual + RAG) ---
else:
    st.title(f"🤖 Chatting in {target_lang}")
    st.caption(f"Knowledge Base: {'Active' if st.session_state.vector_db else 'Empty'}")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Logic
    if prompt := st.chat_input("Ask about your document or say hi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG Retrieval
        context = ""
        if st.session_state.vector_db:
            search_results = st.session_state.vector_db.similarity_search(prompt, k=3)
            context = "\n".join([doc.page_content for doc in search_results])

        with st.chat_message("assistant"):
            # Enhanced RAG + Translation Prompt
            system_instruction = (
                f"You are GlobaLingo. Use the provided context to answer: {context}\n\n"
                f"Regardless of input, you must respond in {target_lang}. "
                "If asked for a summary, provide a concise overview of the document context. "
                "After your response, add '---' and a full English translation."
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_instruction}, *st.session_state.messages]
            )
            
            full_response = response.choices[0].message.content
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
