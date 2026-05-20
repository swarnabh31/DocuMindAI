import streamlit as st
import os
import tempfile
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
DB_FOLDER = "chroma_db_v2"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1:latest"

# --- Core RAG Functions ---

def load_document(file_path):
    """Loads documents based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
    else:
        return []
    return loader.load()

def process_files(uploaded_files):
    """Handles uploaded files and creates the vector store."""
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            docs = load_document(tmp_path)
            documents.extend(docs)
        finally:
            os.remove(tmp_path)

    if not documents:
        return None

    # Advanced Splitting: Use a larger chunk size for better context retention in V2
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Recreate the vector store for each session or keep persist_directory
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_FOLDER
    )
    return vector_store

def get_rag_chain(vector_store, mode="chat"):
    """Creates a RAG chain based on the mode (Chat vs Technical Blog)."""
    llm = ChatOllama(model=LLM_MODEL)
    
    if mode == "Professional":
        system_prompt = (
            "You are a Senior Technical Consultant and Executive Communication Expert. "
            "Your goal is to synthesize the provided context into a high-impact, professional executive report. "
            "CRITICAL FORMATTING REQUIREMENTS:\n"
            "1. VISUAL STRUCTURE: Do not just provide a wall of text. Use a clear hierarchy of information.\n"
            "2. HEADINGS: Use clear, bolded section headers (e.g., **EXECUTIVE SUMMARY**, **TECHNICAL ARCHITECTURE**, **KEY INSIGHTS**).\n"
            "3. EMPHASIS: Use **bolding** for key terms, critical metrics, and essential takeaways to make the document skimmable.\n"
            "4. LISTS: Use a mix of bullet points for features/findings and numbered lists for sequential processes or priority rankings.\n"
            "5. PARAGRAPH FORMATTING: Use short, punchy paragraphs. Every paragraph must be visually distinct and serve a purpose.\n"
            "6. STYLE: Professional, authoritative, and crisp. Avoid generic AI phrases like 'In conclusion' or 'It is important to note'.\n\n"
            "Context: {context}\n\n"
            "Request: {question}"
        )
    else:
        system_prompt = (
            "You are a high-precision Research Assistant. "
            "Your task is to answer the user's question using only the retrieved context. "
            "Guidelines:\n"
            "1. Be comprehensive yet concise. Do not provide one-sentence answers if the context allows for more depth.\n"
            "2. Citation: If the context allows, refer to the specific part of the document.\n"
            "3. Honesty: If the answer is not in the context, explicitly state that you cannot find the information in the provided documents.\n"
            "4. Reasoning: Use your deep reasoning capabilities to connect dots between different chunks of information.\n\n"
            "Context: {context}\n\n"
            "Question: {question}"
        )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # LCEL Pipeline
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": vector_store.as_retriever(search_kwargs={"k": 5}) | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit UI ---

st.set_page_config(page_title="DocuMind AI v2", page_icon="⚡", layout="wide")

st.title("⚡ DocuMind AI v2")
st.markdown("### Chat with your PDFs, Docs, and Text files completely offline.")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.selectbox("Output Mode", ["Chat", "Professional"], index=0)
    st.info("Mode 'Professional' transforms summaries into structured, high-impact reports.")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "docx", "doc", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Analyzing documents..."):
                st.session_state.vector_store = process_files(uploaded_files)
                st.success("Documents indexed successfully!")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.warning("Please upload and process documents in the sidebar to begin.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create chain on the fly based on mode
                current_mode = "Professional" if mode == "Professional" else "chat"
                chain = get_rag_chain(st.session_state.vector_store, mode=current_mode)
                response = chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
