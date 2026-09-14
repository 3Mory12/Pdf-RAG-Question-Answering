# 📄 PDF RAG Question Answering

A simple **Retrieval-Augmented Generation (RAG)** system for answering questions from PDF documents.

The project combines **PDF text extraction, text chunking, semantic embeddings, FAISS vector search, and Mistral Nemo** to retrieve relevant information from a document and generate an answer based on the retrieved context.

## 🚀 Project Overview

The pipeline follows these main steps:

```text
PDF Document
     ↓
Text Extraction
     ↓
Chunking
     ↓
Text Embeddings
     ↓
FAISS Vector Database
     ↓
Question Embedding
     ↓
Similarity Search
     ↓
Relevant Chunks
     ↓
Mistral Nemo
     ↓
Generated Answer
```

## 🧠 Technologies Used

* **Python**
* **PyPDF2** — PDF text extraction
* **Sentence Transformers** — text embeddings
* **FAISS** — vector similarity search
* **Mistral Nemo Instruct** — answer generation
* **Hugging Face Transformers**
* **NumPy**
* **Kaggle** — development environment

## 📦 Installation

Install the required libraries:

```bash
pip install sentence_transformers PyPDF2 faiss_cpu transformers torch
```

## 🔧 How It Works

### 1. Load and Extract PDF Text

The PDF is loaded using `PyPDF2`, and the text from all pages is extracted.

```python
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    return full_text
```

### 2. Text Chunking

The extracted document is divided into smaller chunks to make retrieval more effective.

```python
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
```

### 3. Generate Embeddings

Each text chunk is converted into a numerical vector using:

`sentence-transformers/all-MiniLM-L6-v2`

```python
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

These embeddings represent the semantic meaning of the text.

### 4. Create FAISS Vector Index

FAISS is used to store the embeddings and perform similarity searches.

```python
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index
```

### 5. Retrieve Relevant Context

When a user asks a question, the question is converted into an embedding and compared with the document chunks.

The most relevant chunks are returned:

```python
top_chunks = search_index(
    question,
    model_embeddings,
    index,
    chunks,
    k=3
)
```

### 6. Generate the Answer

The retrieved chunk is passed to **Mistral Nemo Instruct** together with the user's question.

```python
prompt = f"""
Answer the next question: {question}
by reading the following text:
{chunk}
"""
```

The model then generates the final response.

## 📁 Project Structure

```text
pdf-rag-question-answering/
│
├── pdf_rag_question_answering.ipynb
├── README.md
├── requirements.txt
└── data/
    └── sample.pdf
```

> The PDF used during development can be replaced with any text-based PDF document.

## 💡 Example

### Question

```text
In what country and what kind of institution did Whitfield work when he created Python?
```

### Process

The system:

1. Converts the question into an embedding.
2. Searches the FAISS index.
3. Retrieves the most relevant document chunks.
4. Passes the retrieved context to Mistral Nemo.
5. Generates an answer using the provided context.

## ⚠️ Current Limitations

This is a **basic RAG implementation** created for learning and experimentation.

The current version does not yet include a strict mechanism that prevents the language model from answering when the requested information is not present in the retrieved context.

Possible improvements include:

* Context relevance threshold
* Similarity score filtering
* "Information not found" fallback
* Better prompt engineering
* Retrieval from multiple chunks
* Re-ranking retrieved documents
* Source/citation display
* Conversation history
* Support for multiple PDF documents
* Web or UI interface

## 🔮 Future Improvements

Possible future versions could include:

* [ ] Multi-PDF document support
* [ ] Better chunking strategies
* [ ] Metadata filtering
* [ ] Re-ranking
* [ ] Context relevance detection
* [ ] Hallucination prevention
* [ ] Source citations
* [ ] Streamlit or Gradio interface
* [ ] Persistent FAISS index
* [ ] Arabic + English support
* [ ] Chat history
* [ ] API deployment with FastAPI

## 🎯 Purpose

This project was developed as a practical implementation for understanding the fundamental components of **Retrieval-Augmented Generation (RAG)** and how vector databases can be combined with Large Language Models for document-based question answering.

## 📜 License

This project is intended for educational and experimental purposes.
