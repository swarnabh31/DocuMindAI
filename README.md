# DocuMind AI v2 (Professional RAG) ⚡

A professional-grade, offline document intelligence tool. This version transforms your local files into a searchable knowledge base with the ability to generate technical content.

## 🚀 New in Version 2
- **Streamlit UI**: A modern, intuitive web interface.
- **Multi-Format Support**: Now supports `.pdf`, `.docx`, `.doc`, `.txt`, and `.md`.
- **Advanced Prompting**: Specialized system prompts for "Research Assistant" and "Technical Writer" personas.
- **Technical Blog Mode**: A dedicated mode to transform raw summaries into structured, professional blog posts.
- **Enhanced Retrieval**: Increased chunk sizes and higher retrieval counts (`k=5`) for better context window utilization.

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
- Pull the models:
  ```bash
  ollama pull deepseek-r1:latest
  ollama pull nomic-embed-text
  ```

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Running the Application
```bash
streamlit run app.py
```

## 📁 How it Works
1. **Upload**: Drop your files in the sidebar.
2. **Index**: Click 'Process Documents' to create a local vector embedding of your files.
3. **Switch Mode**: Choose between 'Chat' (for precise answers) and 'Technical Blog' (for high-quality long-form content).
4. **Query**: Ask your questions in the chat interface.
