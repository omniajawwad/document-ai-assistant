# 🤖 Document AI Assistant

An AI-powered application that allows users to upload PDF documents and chat with them using **Local RAG (Retrieval-Augmented Generation)** and **Ollama LLMs**.

---

## 🚀 Features

- 📄 Upload any PDF document
- ⚡ Build vector index locally (FAISS)
- 💬 Ask questions about the document
- 🧠 Context-aware answers using RAG
- 🔄 Streaming responses like ChatGPT
- 🔒 Runs locally (No paid API required)

---

## 🧠 How It Works

1. Upload a PDF  
2. The system splits the document into chunks  
3. Creates embeddings using HuggingFace  
4. Stores them in FAISS vector database  
5. Retrieves relevant chunks based on your question  
6. Sends context to LLM (Ollama) to generate answer  

---

## 🛠 Tech Stack

- Python  
- LangChain  
- FAISS  
- HuggingFace Embeddings  
- Ollama (Qwen model)  
- Gradio  

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
