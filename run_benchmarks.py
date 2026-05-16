import json
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import time
import os

# Configure Gemini
API_KEY = "AIzaSyAnqk1drgAlH519ZpNfHNZGNqoWvDPUM9M"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# 1. Pipeline 1: Raw LLM (No RAG)
def pipeline_1_raw_llm(question, context_text=""):
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer based on context if provided, else use your knowledge."
    start = time.time()
    response = model.generate_content(prompt)
    latency = time.time() - start
    # Using simple word count approximation if actual exact token counter is not easily accessible via basic API limits
    return {
        "answer": response.text,
        "latency_sec": latency,
        "tokens_approx": len(prompt.split()) + len(response.text.split())
    }

# 2. Setup Vector DB for Pipeline 2 (Basic RAG)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="arxiv_papers", embedding_function=sentence_transformer_ef)

def ingest_to_basic_rag():
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} items.")
        return
    
    print("Ingesting data into Basic RAG (ChromaDB)...")
    docs = []
    metadatas = []
    ids = []
    
    with open("dataset/materials_science_papers.jsonl", "r", encoding="utf-8") as f:
        # Load first 1000 for Basic RAG benchmark to be fast locally
        for i, line in enumerate(f):
            if i >= 1000: break 
            data = json.loads(line)
            docs.append(data["summary"])
            metadatas.append({"title": data["title"]})
            ids.append(str(i))
            
            if len(docs) == 100:
                collection.add(documents=docs, metadatas=metadatas, ids=ids)
                docs, metadatas, ids = [], [], []
                print(f"Ingested {i+1} documents")
                
    if docs:
        collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"Total ingested: {collection.count()}")

def pipeline_2_basic_rag(question):
    start = time.time()
    
    # Retrieve top 5 matches
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    context = "\n\n".join(results['documents'][0])
    
    prompt = f"Context from Vector Search:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    # Generate Answer
    response = model.generate_content(prompt)
    latency = time.time() - start
    
    return {
        "answer": response.text,
        "latency_sec": latency,
        "tokens_approx": len(prompt.split()) + len(response.text.split()),
        "context_sources": results['metadatas'][0]
    }

if __name__ == "__main__":
    ingest_to_basic_rag()
    
    test_question = "What is the structural evolution of Ti/Cu multilayers based on period thickness?"
    print(f"\n--- Testing Question: {test_question} ---")
    
    print("\n[Pipeline 1: Raw LLM]")
    res1 = pipeline_1_raw_llm(test_question)
    print(f"Tokens: {res1['tokens_approx']}, Latency: {res1['latency_sec']:.2f}s")
    print(f"Answer: {res1['answer'][:150]}...")
    
    print("\n[Pipeline 2: Basic RAG]")
    res2 = pipeline_2_basic_rag(test_question)
    print(f"Tokens: {res2['tokens_approx']}, Latency: {res2['latency_sec']:.2f}s")
    print(f"Answer: {res2['answer'][:150]}...")
