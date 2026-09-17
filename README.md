# 📚 Research Paper Assistant

An AI-powered research paper assistant built with **Streamlit**, **ChromaDB**, **Sentence Transformers**, and **Groq LLM**. Upload academic PDFs and interact with them through natural language — chat, analyze, and compare papers effortlessly.

---

## 📹 Demo Video

Watch the 3-minute application walkthrough:
👉 [Click Here to Watch the Demo Video](demo_video.mp4)

---

## ✨ Features

- 📤 **Multi-Paper Upload** — Upload and process multiple research PDFs at once
- 💬 **Agent Chat** — Conversational QA with memory, grounded in your papers with source citations
- 📊 **Document Analysis** — One-click generation of summaries, key contributions, limitations, and future work
- ⚖️ **Paper Comparison** — Side-by-side comparison of 2+ papers on any topic or question
- 🔍 **RAG Pipeline** — Retrieval-Augmented Generation using ChromaDB vector store + Sentence Transformers embeddings

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/agrawaladiti33-boop/Research_Paper_Assistant.git
cd Research_Paper_Assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your [Groq API key](https://console.groq.com/):
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🏗️ Project Structure

```
Research_Paper_Assistant/
├── app.py                  # Main Streamlit application
├── document_processor.py   # PDF loading and chunking
├── vector_manager.py       # ChromaDB vector store management
├── llm_manager.py          # Groq LLM interaction
├── embeddings.py           # Sentence Transformer embeddings
├── prompts.py              # LLM prompt templates
├── rag.py                  # RAG pipeline logic
├── retrieval.py            # Retrieval utilities
├── analysis.py             # Document analysis helpers
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variable template
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Vector DB | ChromaDB |
| Embeddings | Sentence Transformers |
| LLM | Groq (Llama / Mixtral) |
| PDF Parsing | PyPDF |
| RAG Framework | LangChain |

---

## ⚙️ Configuration

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com/) |

---

## 📝 License

MIT License — feel free to use, modify, and distribute.
