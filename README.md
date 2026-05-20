# DocuMind AI (Professional RAG) ⚡

A professional-grade, offline document intelligence tool. DocuMind AI transforms your local files into a searchable knowledge base, allowing you to query complex documents or synthesize them into high-impact professional reports.

## 🚀 Key Features
- **Multi-Format Support**: Seamlessly handles `.pdf`, `.docx`, `.doc`, `.txt`, and `.md`.
- **Professional Personas**: 
    - **Chat Mode**: High-precision research assistant for accurate, context-aware answers.
    - **Professional Mode**: Transforms raw data into structured, executive-style reports with bolded headers and key insights.
- **100% Offline & Private**: Powered by Ollama, ensuring your data never leaves your machine.
- **Advanced Retrieval**: Optimized chunking and `k=5` retrieval for superior context synthesis.

## 🛠 Tech Stack
- **Frontend**: Streamlit
- **LLM**: Ollama (`deepseek-r1:latest`)
- **Embeddings**: Ollama (`nomic-embed-text`)
- **Orchestration**: LangChain (LCEL)
- **Vector Store**: ChromaDB
- **Loaders**: PyPDF, Unstructured, TextLoader

## 🏁 Getting Started

### 1. Prerequisites
- Install [Ollama](https://ollama.ai/).
- Pull the required models:
  ```bash
  ollama pull deepseek-r1:latest
  ollama pull nomic-embed-text
  ```

### 2. Installation
```bash
pip install -r requirements.txt
```

## 🕹️ Usage Options

You can interact with the tool in two ways: via the professional Web UI or the lightweight CLI.

### Option A: The Professional Web UI (Recommended)
The modern interface for a seamless, high-impact experience.
```bash
streamlit run app.py
```
**Workflow:**
1. **Upload**: Drop your files in the sidebar.
2. **Index**: Click 'Process Documents' to create your local vector knowledge base.
3. **Switch Mode**: Choose between **Chat** and **Professional** in the dropdown.
4. **Query**: Start chatting with your documents.

### Option B: The Lightweight CLI
Ideal for quick tests or integration into other scripts.
1. **Ingest Documents**:
   - Place your PDFs/files in your source directory.
   - Run: `python ingest.py`
2. **Chat**:
   - Run: `python chat.py`

## 📁 Project Structure
- `app.py`: The main Streamlit application (V2).
- `ingest.py`: CLI script to process and embed documents.
- `chat.py`: CLI script for interactive querying.
- `requirements.txt`: Project dependencies.
- `.gitignore`: Ensures local databases and secrets aren't uploaded.
