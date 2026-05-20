import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
DB_FOLDER = "./chroma_db"
LLM_MODEL = "deepseek-r1:latest"

def chat_with_pdfs():
    """
    Loads the vector store and allows the user to ask questions about the PDFs.
    Using an LCEL chain to avoid 'langchain.chains' import issues.
    """
    if not os.path.exists(DB_FOLDER):
        print("Vector store not found. Please run ingest.py first.")
        return

    # Initialize Ollama Embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Load the existing vector store
    vector_store = Chroma(
        persist_directory=DB_FOLDER, 
        embedding_function=embeddings
    )

    # Initialize the LLM
    llm = ChatOllama(model=LLM_MODEL)

    # Define the system prompt
    template = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)

    # Create a manual RAG chain using LCEL (LangChain Expression Language)
    # This avoids relying on the 'langchain.chains' module entirely
    retriever = vector_store.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The Chain: 
    # 1. Retrieve docs -> 2. Format them -> 3. Pass to Prompt -> 4. LLM -> 5. Parse to String
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

    print("\n--- Offline PDF Chatbot Ready (LCEL Version) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Ask a question about your PDFs: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query.strip():
            continue

        print("\nThinking...")
        try:
            response = rag_chain.invoke(query)
            print(f"\nAnswer: {response}\n")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    chat_with_pdfs()
