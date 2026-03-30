import numpy as np
import faiss
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

PDF_PATH = "The Intelligent Investor.pdf"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 7

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    return np.array(embedding_model.encode(chunks))

def create_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, k=3):
    query_emb = embedding_model.encode([query])
    D, I = index.search(np.array(query_emb), k)
    return [chunks[i] for i in I[0]]

def generate_answer(query, context):
    prompt = f"""
You are an expert assistant.

Rules:
- Answer ONLY using the context
- Combine information from multiple parts
- Explain clearly in 2–4 sentences
- If not found, say: "Not found in document"

Context:
{context}

Question:
{query}
"""

    response = model.generate_content(prompt)
    return response.text

def main():
    print("\n" + "="*70)
    print("RAG SYSTEM - STOCK MARKET & INVESTMENT ANALYSIS")
    print("="*70)
    
    # Step 1: Load PDF
    print("\n[STEP 1] Loading PDF...")
    text = load_pdf(PDF_PATH)
    print(f"✓ PDF loaded successfully. Total characters: {len(text)}")

    # Step 2: Chunk text
    print("\n[STEP 2] Chunking text...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"✓ Text chunked successfully. Total chunks: {len(chunks)}")

    # Step 3: Create embeddings
    print("\n[STEP 3] Creating embeddings...")
    embeddings = create_embeddings(chunks)
    print(f"✓ Embeddings created. Shape: {embeddings.shape}")

    # Step 4: Create FAISS index
    print("\n[STEP 4] Creating FAISS index...")
    index = create_index(embeddings)
    print(f"✓ FAISS index created successfully")

    # Step 5: Display sample chunks
    print("\n" + "="*70)
    print("[BACKEND VERIFICATION] SAMPLE CHUNKS")
    print("="*70)
    for i in range(2):
        print(f"\n--- CHUNK {i} ---")
        print(chunks[i][:400])
        print("...")

    # Step 6: Display sample embeddings
    print("\n" + "="*70)
    print("[BACKEND VERIFICATION] SAMPLE EMBEDDINGS")
    print("="*70)
    for i in range(2):
        print(f"\n--- EMBEDDING {i} (First 15 values) ---")
        print(embeddings[i][:15])
        print(f"Full embedding shape: {embeddings[i].shape}")

    # Step 7: Run queries
    questions = [
        "how to deal with brokerage houses?",
        "what is theory of diversification?",
        "how to become intelligent investor?",
        "how to do business valuation?",
        "what is putting all eggs in one basket analogy?"
    ]

    print("\n" + "="*70)
    print("[LIVE QUERYING] RUNNING MANDATORY QUERIES")
    print("="*70)

    for idx, q in enumerate(questions, 1):
        print("\n" + "-"*70)
        print(f"QUERY {idx}/{len(questions)}")
        print("-"*70)
        print(f"\nQUESTION: {q}\n")

        relevant_chunks = retrieve(q, index, chunks, TOP_K)
        context = "\n".join(relevant_chunks)

        print("RETRIEVED CONTEXT (showing first 800 characters):")
        print("-"*70)
        print(context[:800])
        print("...")

        answer = generate_answer(q, context)

        print("\nANSWER:")
        print("-"*70)
        print(answer)
        print("\n")
if __name__ == "__main__":
    main()