import arxiv
import tiktoken
import json
import os
import time

enc = tiktoken.get_encoding("cl100k_base")

def fetch_arxiv_papers(query="cat:cond-mat.mtrl-sci", max_results=1000):
    client = arxiv.Client(page_size=1000, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance # Use relevance for targeted queries
    )
    
    total_tokens = 0
    papers = []
    
    print(f"Fetching up to {max_results} papers for query: {query}")
    try:
        for result in client.results(search):
            text = f"{result.title}\n{result.summary}\n"
            tokens = len(enc.encode(text))
            total_tokens += tokens
            
            papers.append({
                "id": result.entry_id,
                "title": result.title,
                "published": str(result.published),
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "categories": result.categories
            })
    except Exception as e:
        print(f"Error fetching for query {query}: {e}")

    return papers, total_tokens

# 1. Load Ground Truth Questions to build targeted queries
with open("ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# Extract simple keywords by removing common words (very basic extraction)
stop_words = {"what", "is", "the", "of", "in", "how", "does", "are", "which", "and", "to", "on", "for", "with", "a", "an", "typically", "reported", "recent", "literature"}
targeted_queries = []
for item in ground_truth:
    words = [w.strip("?") for w in item["question"].lower().split() if w.lower() not in stop_words and len(w) > 3]
    if words:
        # Build an arxiv query, e.g., "all:gallium AND all:nitride"
        query = " AND ".join([f"all:{w}" for w in words[:4]])
        targeted_queries.append(query)

all_papers = {}
total_tokens = 0

print("--- Step 1: Fetching Targeted Papers ---")
for query in targeted_queries:
    papers, tokens = fetch_arxiv_papers(query=query, max_results=50) # Get top 50 relevant papers per question
    for p in papers:
        if p["id"] not in all_papers:
            all_papers[p["id"]] = p
            total_tokens += len(enc.encode(p["title"] + " " + p["summary"]))
    time.sleep(1)

print(f"Targeted papers collected: {len(all_papers)}")

print("\n--- Step 2: Fetching Random Papers (Padding the Haystack) ---")
# Need roughly 10,000 papers total
remaining = max(0, 10000 - len(all_papers))
if remaining > 0:
    # Use SubmittedDate to get recent random papers
    client = arxiv.Client(page_size=1000, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query="cat:cond-mat.mtrl-sci",
        max_results=remaining,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    for result in client.results(search):
        if result.entry_id not in all_papers:
            text = f"{result.title}\n{result.summary}\n"
            tokens = len(enc.encode(text))
            total_tokens += tokens
            all_papers[result.entry_id] = {
                "id": result.entry_id,
                "title": result.title,
                "published": str(result.published),
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "categories": result.categories
            }
        if len(all_papers) % 500 == 0:
            print(f"Total papers collected: {len(all_papers)}...")

print(f"\nTotal papers collected: {len(all_papers)}")
print(f"Total tokens collected: {total_tokens}")

os.makedirs('dataset', exist_ok=True)
with open('dataset/materials_science_papers.jsonl', 'w', encoding='utf-8') as f:
    for paper in all_papers.values():
        f.write(json.dumps(paper) + '\n')
        
print("Saved to dataset/materials_science_papers.jsonl")