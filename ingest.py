import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
PDF_FOLDER = "./pdfs"
DB_FOLDER = "./chroma_db"

def process_pdfs():
    """
    Reads all PDFs from the PDF_FOLDER, splits them into chunks, 
    and stores them in a ChromaDB vector store.
    """
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created {PDF_FOLDER} folder. Please drop your PDFs there and run again.")
        return False

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the pdfs folder.")
        return False

    print(f"Found {len(pdf_files)} PDF(s). Processing...")

    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, pdf))
        documents.extend(loader.load())

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Initialize Ollama Embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_FOLDER
    )
    
    print(f"Successfully processed {len(chunks)} chunks into the vector store.")
    return True

if __name__ == "__main__":
    process_pdfs()
