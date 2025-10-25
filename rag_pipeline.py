import os
import sys
import fitz  # PyMuPDF
import json
import faiss
import numpy as np
import nltk
import ollama

from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer data
nltk.download('punkt')
nltk.download('punkt-tab')

# Configuration
index_path = "faiss_index.index"
chunks_path = "chunks.json"
chunk_char_limit = 1000
chunk_overlap = 200
top_k = 5
embedding_model = "nomic-embed-text"
llm_model = "llama3"

def extract_text_from_pdfs(folder_path):
    all_texts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            all_texts.append(text)
    return all_texts

def semantic_chunk_texts(texts, chunk_char_limit=1000, chunk_overlap=200):
    chunks = []
    for text in texts:
        sentences = sent_tokenize(text)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_char_limit:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())

    # Add overlap
    overlapped_chunks = []
    for i in range(len(chunks)):
        chunk = chunks[i]
        if i > 0:
            overlap = chunks[i - 1][-chunk_overlap:]
            chunk = overlap + " " + chunk
        overlapped_chunks.append(chunk.strip())
    return overlapped_chunks

def embed_chunks_ollama(chunks, model_name, max_chars=8192):
    embeddings = []
    for i, chunk in enumerate(chunks):
        truncated_chunk = chunk[:max_chars]
        try:
            response = ollama.embeddings(model=model_name, prompt=truncated_chunk)
            embeddings.append(response['embedding'])
        except Exception as e:
            print(f"⚠️ Skipping chunk {i} due to error: {e}")
    return embeddings

def store_embeddings_faiss(embeddings, index_path):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, index_path)

def save_chunks(chunks, chunks_path):
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)

def load_index_and_chunks(index_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    return index, chunks

def embed_query_ollama(query, model_name):
    response = ollama.embeddings(model=model_name, prompt=query)
    return response['embedding']

def query_with_ollama(query, index, chunks, embedding_model, llm_model, top_k=5):
    query_embedding = embed_query_ollama(query, embedding_model)
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

def main():
    if len(sys.argv) != 2:
        print("Usage: python rag_pipeline.py <pdf_folder_path>")
        sys.exit(1)

    pdf_folder = sys.argv[1]

    if not os.path.isdir(pdf_folder):
        print(f"Error: '{pdf_folder}' is not a valid directory.")
        sys.exit(1)

    print("Extracting text from PDF files...")
    texts = extract_text_from_pdfs(pdf_folder)

    print("Performing semantic-aware chunking...")
    chunks = semantic_chunk_texts(texts, chunk_char_limit, chunk_overlap)

    print("Embedding chunks with Ollama...")
    embeddings = embed_chunks_ollama(chunks, embedding_model)

    print("Storing embeddings in FAISS index...")
    store_embeddings_faiss(embeddings, index_path)

    print("Saving chunks to JSON file...")
    save_chunks(chunks, chunks_path)

    print(f"✅ FAISS index saved to {index_path}")
    print(f"✅ Text chunks saved to {chunks_path}")

    print("\n🔍 Ready for RAG-style Q&A with Ollama")
    index, chunks = load_index_and_chunks(index_path, chunks_path)

    while True:
        query = input("Enter your question (or type 'exit()' to quit): ")
        if query.strip().lower() == "exit()":
            print("👋 Exiting RAG pipeline.")
            break
        answer = query_with_ollama(query, index, chunks, embedding_model, llm_model, top_k)
        print("\nLLM Response:\n")
        print(answer)

if __name__ == "__main__":
    main()

