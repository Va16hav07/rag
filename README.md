# RAG System - Stock Market & Investment Analysis

## Project Overview

A **Retrieval-Augmented Generation (RAG) System** that ingests investment textbooks and answers financial questions using semantic search and generative AI. This project demonstrates the end-to-end RAG pipeline: from PDF ingestion and vector embeddings to semantic retrieval and response generation.

## Assignment Details

**Course:** Stock Market & Investment Analysis  
**Instructor:** Achint Setia  
**Date:** March 5, 2026  
**Topic:** Retrieval-Augmented Generation Implementation

### Assignment Objective

Build and demonstrate a functional RAG system using *The Intelligent Investor* textbook that:
- Ingests PDF documents and creates text chunks
- Generates semantic embeddings for chunks
- Retrieves relevant content using vector similarity search
- Generates accurate answers using Google Gemini AI

---

## Key Features

✅ **PDF Processing** - Extracts and chunks text from investment books  
✅ **Semantic Embeddings** - Uses SentenceTransformer for 384-dimensional vectors  
✅ **Fast Retrieval** - FAISS vector database for efficient similarity search  
✅ **Generative AI** - Google Gemini API for context-aware answer generation  
✅ **Backend Verification** - Displays sample chunks and embeddings  
✅ **Mandatory Query Testing** - Demonstrates 5 specific investment questions  

---

## System Architecture

```
PDF Document
    ↓
Text Chunking (800 chars + 100 char overlap)
    ↓
Semantic Embeddings (SentenceTransformer)
    ↓
FAISS Vector Index
    ↓
Query Processing
    ↓
Top-K Retrieval (k=7)
    ↓
Context + Prompt → Gemini AI
    ↓
Answer Generation
```

---

## Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Google API Key for Gemini AI
- *The Intelligent Investor.pdf* file

---

## Installation

### 1. Clone/Setup Project
```bash
cd /home/freshuser/Desktop/rag\ assignment
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variable
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

---

## Usage

### Run the RAG System
```bash
python main.py
```

### Expected Output
- System initialization progress
- Sample text chunks display
- Sample embedding vectors
- Answers to 5 mandatory questions:
  1. How to deal with brokerage houses?
  2. What is theory of diversification?
  3. How to become intelligent investor?
  4. How to do business valuation?
  5. What is putting all eggs in one basket analogy?

---

## Project Structure

```
rag assignment/
├── main.py                          # Main RAG system
├── app.py                           # Alternative version
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── The Intelligent Investor.pdf     # Source document
└── venv/                            # Virtual environment
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `faiss-cpu` | Vector similarity search |
| `PyPDF2` | PDF text extraction |
| `sentence-transformers` | Semantic embedding generation |
| `google-generativeai` | Gemini API integration |

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 800 | Characters per text chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between consecutive chunks |
| `TOP_K` | 7 | Number of chunks to retrieve per query |
| `EMBEDDING_DIM` | 384 | Dimension of semantic embeddings |
| `MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `LLM` | `gemini-2.5-flash` | Google's generative model |

---

## Video Submission Requirements

### Recording Checklist
- ✅ Webcam visible during entire recording
- ✅ Show system initialization (Steps 1-4)
- ✅ Display 2 sample chunks with explanations
- ✅ Show 2 sample embeddings with dimension info
- ✅ Run all 5 mandatory questions
- ✅ Scroll slowly through answers (legible text)
- ✅ Show completion message

### File Naming
```
[YourName]_[YourRollNumber].mp4
```

### Upload
Submit to the designated Google Drive folder for the course.

---

## Data Privacy & Ethics

⚠️ **Important Notices:**
- Source material is for **personal educational use only**
- **Do not distribute** the book or generated database outside this course
- Ensure chunks and embeddings **accurately represent** the source text
- **No hallucination** - all answers grounded in source document

---

## How the RAG Pipeline Works

### 1. **Ingestion**
- PDF is read using PyPDF2
- Text is extracted page by page

### 2. **Chunking**
- Text split into 800-character chunks
- 100-character overlap maintains context between chunks
- Total: ~1,837 chunks from the book

### 3. **Embedding**
- Each chunk encoded to 384-dimensional vector
- Uses SentenceTransformer's `all-MiniLM-L6-v2` model
- Captures semantic meaning of text

### 4. **Indexing**
- FAISS creates efficient L2-distance index
- Enables fast similarity search

### 5. **Retrieval**
- User query converted to embedding
- Top-7 similar chunks retrieved using L2 distance
- Combined as context

### 6. **Generation**
- Context + Query sent to Gemini AI
- Model generates answer grounded in context
- System prompts enforce document-only response

---

## Troubleshooting

### Missing API Key Error
```bash
ValueError: GOOGLE_API_KEY environment variable not set
```
**Solution:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### PDF Not Found
Ensure `The Intelligent Investor.pdf` is in the project root directory.

### Memory Issues
If system runs out of memory, reduce `CHUNK_SIZE` or process PDF in parts.

### Slow Embedding Generation
First run generates embeddings (takes ~2-3 minutes). Subsequent runs will be faster once index is built.

---

## Performance Metrics

- **PDF Loading:** ~2-3 seconds
- **Chunking:** <1 second (1,837 chunks)
- **Embedding Generation:** ~60-90 seconds (first run)
- **FAISS Index Creation:** <1 second
- **Per Query Processing:** ~3-5 seconds

---

## Future Enhancements

- 🔄 Persistent embedding storage (pickle/database)
- 🗄️ Support for multiple PDFs
- 🎯 Fine-tuned embeddings for finance domain
- 💾 Firebase/MongoDB backend
- 🌐 Web interface (Flask/FastAPI)
- 📊 Query performance metrics
- 🔐 API key encryption

---

## References

- **The Intelligent Investor** - Benjamin Graham (Revised Edition)
- FAISS: https://github.com/facebookresearch/faiss
- SentenceTransformers: https://www.sbert.net/
- Google Generative AI: https://ai.google.dev/

---

## Credits

**Original Concept & Implementation:** [@manit101](https://github.com/manit101)


---

## License

This project is for educational purposes as part of the course assignment on Retrieval-Augmented Generation systems. 

**Copyright Notice:** The source text (*The Intelligent Investor*) is protected by copyright and used for educational purposes only under fair use guidelines.

---
