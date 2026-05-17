"""
Pipeline 3 — TigerGraph GraphRAG

Uses TigerGraph Cloud 'Crystal' graph. On first run, upserts materials-science
papers from the local JSONL into TigerGraph as Papers vertices (with categories
and author metadata). Then performs multi-hop graph traversal:

  Seed Papers -> (shared category edges) -> Related Papers -> Authors

and generates an answer with Gemini using the graph-augmented context.

This is TRUE GraphRAG: retrieval is driven by graph structure, not just
vector similarity.
"""

import json
import os
import re
import sys
import time
import io
import concurrent.futures

import requests
import google.generativeai as genai
from pyTigerGraph import TigerGraphConnection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

TIGERGRAPH_HOST   = os.getenv("TG_HOST",   "https://tg-13e1516a-fc0d-4add-8cd9-160f98c22473.tg-2635877100.i.tgcloud.io")
TIGERGRAPH_SECRET = os.getenv("TG_SECRET")
GRAPH_NAME        = os.getenv("TG_GRAPH",  "Crystal")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
LLM = genai.GenerativeModel("gemini-2.5-flash")

MATERIALS_DATASET = os.getenv("DATASET_PATH", "dataset/materials_science_papers.jsonl")
INGEST_LIMIT      = int(os.getenv("INGEST_LIMIT", "1000"))  # papers to upsert
MATERIALS_PREFIX  = "mat_"                                   # vertex ID prefix for our data

# Module-level cache to avoid re-auth and re-fetching on every query
_cached_token = None
_cached_conn  = None
_cached_papers = None  # cached list of all material papers from TigerGraph

QUESTION = os.getenv(
    "QUESTION",
    "What is the structural evolution of Ti/Cu multilayers based on period thickness?",
)

# ---------------------------------------------------------------------------
# TigerGraph auth
# ---------------------------------------------------------------------------

def get_token() -> str:
    r = requests.post(
        f"{TIGERGRAPH_HOST}/gsql/v1/tokens",
        json={"secret": TIGERGRAPH_SECRET, "graph": GRAPH_NAME},
        timeout=20,
    )
    if r.status_code != 200:
        raise SystemExit(f"[FATAL] Token request failed ({r.status_code}): {r.text[:300]}")
    data = r.json()
    print(f"  JWT obtained (expires {data.get('expiration','?')})")
    return data["token"]


def connect(token: str) -> TigerGraphConnection:
    return TigerGraphConnection(
        host=TIGERGRAPH_HOST,
        graphname=GRAPH_NAME,
        gsPort="443",
        restppPort="443",
        tgCloud=True,
        apiToken=token,
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the","and","with","from","what","based","how","does","which","are","for",
    "this","that","their","its","has","have","been","can","will","into","over",
    "not","also","was","were","but","than","more","all","one","two","three",
    "new","use","used","using","show","shows","shown","present","study",
    "studies","paper","propose","proposed","result","results","method",
    "methods","approach","between","both","these","such","they","our","we",
    "an","in","of","is","a","to","on","at","by","be","it","via","as","an",
    "are","due","our","here","then","when","where","each","very","high",
    "low","large","small","well","may","also","its","within","however",
    "thus","while","since","under","up","down","about","among","upon",
}

def tokenize(text: str) -> set:
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in STOPWORDS
    }


# ---------------------------------------------------------------------------
# Step 1: Ingest materials dataset into TigerGraph
# ---------------------------------------------------------------------------

def ingest_materials(conn: TigerGraphConnection) -> int:
    """
    Upsert materials-science papers into the 'Papers' vertex type.
    Skips if already loaded (checks for our prefix in vertex IDs).
    """
    print(f"  Checking existing materials data in TigerGraph ...")
    try:
        total = conn.getVertexCount("Papers")
        if total > 50:
            print(f"  Materials already loaded. Total Papers: {total}")
            return total
    except Exception as e:
        print(f"  Error checking existing vertices: {e}")

    if not os.path.exists(MATERIALS_DATASET):
        print(f"  WARNING: Dataset file not found: {MATERIALS_DATASET}")
        return 0

    print(f"  Upserting up to {INGEST_LIMIT} materials papers into TigerGraph ...")
    count = 0
    errors = 0

    with open(MATERIALS_DATASET, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if count >= INGEST_LIMIT:
                break
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            title   = doc.get("title", "")[:500]
            summary = doc.get("summary", "")[:2000]
            arxiv_id = doc.get("id", f"mat_{i}")
            # Derive a category from the arxiv ID prefix if possible
            # e.g. "cond-mat.mtrl-sci" for materials papers
            cats = doc.get("categories", "cond-mat.mtrl-sci")
            year_str = str(doc.get("published", "2020"))[:4]

            vertex_id = f"{MATERIALS_PREFIX}{i}"

            # Fallback parsing if fetch_arxiv wasn't updated
            authors_list = doc.get("authors", [])
            categories_list = doc.get("categories", [cats])
            if isinstance(categories_list, str): categories_list = [categories_list]
            if isinstance(authors_list, str): authors_list = [authors_list]

            try:
                # Upsert Paper Vertex
                conn.upsertVertex("Papers", vertex_id, {
                    "id":       vertex_id,
                    "title":    title,
                    "abstract": summary[:500],
                    "categories": ", ".join(categories_list),
                    "year":     year_str,
                    "column_":  summary[500:900] if len(summary) > 500 else "",
                    "column_2": summary[900:1300] if len(summary) > 900 else "",
                    "column_3": summary[1300:1700] if len(summary) > 1300 else "",
                    "column_4": summary[1700:2000] if len(summary) > 1700 else "",
                    "column_5": ", ".join(categories_list),
                    "column_6": year_str,
                })
                
                # Build Graph Edges
                for cat in categories_list:
                    # Clean Category ID
                    cat_id = re.sub(r'[^a-zA-Z0-9_-]', '_', cat)
                    conn.upsertVertex("Categories", cat_id, {"id": cat_id})
                    conn.upsertEdge("Papers", vertex_id, "Paper_to_Categories", "Categories", cat_id)
                    conn.upsertEdge("Categories", cat_id, "Categories_to_Papers", "Papers", vertex_id)
                    
                    for author in authors_list:
                        # Clean Author ID
                        author_id = re.sub(r'[^a-zA-Z0-9_-]', '_', author)
                        conn.upsertVertex("Authors", author_id, {"id": author_id})
                        conn.upsertEdge("Categories", cat_id, "Categories_to_Authors", "Authors", author_id)
                
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Upsert error on {vertex_id}: {e}")
                continue

    print(f"  Upserted {count} materials papers ({errors} errors)")
    return count


# ---------------------------------------------------------------------------
# Step 2: Retrieve seed papers from TigerGraph
# ---------------------------------------------------------------------------

def find_seed_papers(conn: TigerGraphConnection, question: str, top_k: int = 8) -> list:
    """
    Fetch Papers vertices and score by keyword overlap with the question.
    Returns the top_k most relevant papers.
    """
    q_tokens = tokenize(question)
    print(f"  Query tokens ({len(q_tokens)}): {sorted(q_tokens)[:15]}")

    global _cached_papers

    # Use cached papers if available (avoids re-fetching 10K papers every query)
    if _cached_papers is not None:
        print(f"  Using cached papers ({len(_cached_papers)} papers) ...")
        papers = _cached_papers
    else:
        print(f"  Fetching materials Papers from TigerGraph (first time, caching) ...")
        papers = []
        try:
            # Fetch a large batch of papers and filter locally for our materials dataset
            all_papers = conn.getVertices("Papers", limit=10000)
            for p in all_papers:
                if p["v_id"].startswith("mat_"):
                    attrs = p.get("attributes", {})
                    title = attrs.get("title", "")
                    abstract_parts = [
                        attrs.get("abstract", ""), attrs.get("categories", ""),
                        attrs.get("year", ""), attrs.get("column_", ""),
                        attrs.get("column_2", ""), attrs.get("column_3", ""),
                        attrs.get("column_4", ""),
                    ]
                    full_text = title + " " + " ".join(pt for pt in abstract_parts if pt)
                    p["_precomputed_tokens"] = tokenize(full_text)
                    papers.append(p)
            _cached_papers = papers
            print(f"  Cached {len(papers)} papers for future queries.")
        except Exception as e:
            print(f"  Error fetching Papers: {e}")
            return []

    scored = []
    for p in papers:
        attrs = p.get("attributes", {})
        title = attrs.get("title", "")

        # Reassemble full abstract from split columns
        abstract_parts = [
            attrs.get("abstract", ""),
            attrs.get("categories", ""),
            attrs.get("year", ""),
            attrs.get("column_", ""),
            attrs.get("column_2", ""),
            attrs.get("column_3", ""),
            attrs.get("column_4", ""),
        ]
        
        if "_precomputed_tokens" in p:
            p_tokens = p["_precomputed_tokens"]
        else:
            full_text = title + " " + " ".join(pt for pt in abstract_parts if pt)
            p_tokens = tokenize(full_text)

        overlap = len(q_tokens & p_tokens)
        if overlap >= 2:  # require at least 2 matching tokens for quality
            scored.append({
                "score":    overlap,
                "v_id":     p["v_id"],
                "title":    title.strip(),
                "abstract": " ".join(part for part in abstract_parts if part)[:2000],
                "cats":     attrs.get("column_5", attrs.get("categories", "")),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]
    print(f"  Found {len(scored)} matching papers (>= 2 tokens) — using top {len(top)}")
    for s in top:
        print(f"    [score={s['score']}] {s['v_id']}: {s['title'][:70]}")
    return top


# ---------------------------------------------------------------------------
# Step 3: Multi-hop graph traversal
# ---------------------------------------------------------------------------

def graph_traversal(conn: TigerGraphConnection, seed_papers: list, question: str) -> dict:
    """
    Multi-hop graph traversal:
      Seed Papers --(Paper_to_Categories)--> Categories
                  --(Categories_to_Papers)--> Related Papers
                  --(Categories_to_Authors)--> Authors
    """
    q_tokens = tokenize(question)
    all_categories = set()
    related_paper_ids = []
    all_authors = set()

    # Hop 1: Seed Papers -> Categories (via edges + inline category text)
    for paper in seed_papers:
        pid = paper["v_id"]

        # Try edge traversal
        try:
            cat_edges = conn.getEdges("Papers", pid, edgeType="Paper_to_Categories")
            for edge in cat_edges:
                cat_id = edge.get("to_id", "")
                if cat_id:
                    all_categories.add(cat_id)
        except Exception:
            pass

        # Also parse inline category from attributes
        raw_cats = paper.get("cats", "")
        if raw_cats:
            for cat in re.split(r"[\s,;]+", raw_cats):
                cat = cat.strip()
                if len(cat) > 1:
                    all_categories.add(cat)

    print(f"  Hop 1 -> Categories discovered: {sorted(all_categories)[:12]}")

    # Hop 2: Categories -> Related Papers + Authors
    def fetch_category_edges(cat_id):
        rel_ids = []
        auth_ids = []
        try:
            rel_edges = conn.getEdges("Categories", cat_id, edgeType="Categories_to_Papers")
            rel_ids = [edge.get("to_id", "") for edge in rel_edges[:20] if edge.get("to_id", "")]
        except Exception:
            pass

        try:
            auth_edges = conn.getEdges("Categories", cat_id, edgeType="Categories_to_Authors")
            auth_ids = [edge.get("to_id", "") for edge in auth_edges[:15] if edge.get("to_id", "")]
        except Exception:
            pass
        return rel_ids, auth_ids

    top_categories = list(all_categories)[:6]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = executor.map(fetch_category_edges, top_categories)
        for rel_ids, auth_ids in results:
            for rpid in rel_ids:
                if not any(p["v_id"] == rpid for p in seed_papers):
                    related_paper_ids.append(rpid)
            all_authors.update(auth_ids)

    print(f"  Hop 2 -> Related paper IDs: {len(related_paper_ids)}, Authors: {len(all_authors)}")

    # Fetch related papers and re-score them
    scored_related = []
    unique_related = list(dict.fromkeys(related_paper_ids))[:60]
    if unique_related:
        try:
            batch = conn.getVerticesById("Papers", unique_related[:30])
            for vdata in batch:
                attrs = vdata.get("attributes", {})
                title = attrs.get("title", "")
                abstract_parts = [
                    attrs.get("abstract", ""), attrs.get("categories", ""),
                    attrs.get("column_", ""), attrs.get("column_2", ""),
                ]
                full_text = title + " " + " ".join(p for p in abstract_parts if p)
                r_tokens = tokenize(full_text)
                overlap = len(q_tokens & r_tokens)
                scored_related.append({
                    "v_id":     vdata["v_id"],
                    "title":    title.strip(),
                    "abstract": " ".join(p for p in abstract_parts if p)[:800],
                    "score":    overlap,
                    "hop":      2,
                })
        except Exception as e:
            print(f"  Warning fetching related papers: {e}")

    scored_related.sort(key=lambda x: x["score"], reverse=True)

    return {
        "seed_papers":    seed_papers,
        "categories":     sorted(all_categories),
        "related_papers": scored_related[:6],
        "authors":        sorted(all_authors)[:20],
    }


# ---------------------------------------------------------------------------
# Step 4: Answer generation with Gemini
# ---------------------------------------------------------------------------

API_KEYS = [
    os.getenv("GOOGLE_API_KEY")
]
current_key_idx = 0

def generate_answer(question: str, subgraph: dict) -> dict:
    """Generate a GraphRAG answer using Gemini with multi-hop graph context."""
    global current_key_idx

    seed_ctx = []
    for i, p in enumerate(subgraph["seed_papers"], 1):
        seed_ctx.append(
            f"[Paper {i} | TigerGraph ID: {p['v_id']} | Score: {p['score']}]\n"
            f"Title: {p['title']}\n"
            f"Abstract: {p['abstract'][:1200]}"
        )

    related_ctx = []
    for p in subgraph["related_papers"]:
        if p["title"]:
            related_ctx.append(
                f"[Related | ID: {p['v_id']} | Hop-2 Score: {p['score']}]\n"
                f"Title: {p['title']}\n"
                f"Abstract: {p['abstract'][:600]}"
            )

    cats_str    = ", ".join(subgraph["categories"][:15]) or "N/A"
    authors_str = ", ".join(subgraph["authors"][:12])   or "N/A"

    prompt = f"""You are a materials science expert. Answer the question using the context retrieved from a TigerGraph knowledge graph via multi-hop traversal. If the context does not contain the answer, you may use your general knowledge, but state that the information was not in the retrieved graph.

## TigerGraph GraphRAG Context

### Directly Matched Papers (Hop 1 — keyword retrieval from TigerGraph):
{chr(10).join(seed_ctx)[:3000]}

### Graph-Traversal Related Papers (Hop 2 — via shared Category edges):
{chr(10).join(related_ctx)[:1500] if related_ctx else "(No related papers found via graph traversal)"}

### Research Categories Traversed in Graph:
{cats_str}

### Authors Discovered via Graph (Categories -> Authors edges):
{authors_str}

## Question:
{question}

## Instructions:
- You are an expert materials scientist. Provide a comprehensive, well-explained scientific answer to the question.
- Write your answer as a confident, standalone textbook explanation.
- Present all facts as your own knowledge. NEVER refer to the "provided context", "retrieved text", or "information provided". Do not mention if information is missing; simply answer the question directly using your scientific expertise.

## Answer:"""

    start = time.time()
    
    answer = "Error generating answer."
    try:
        from groq import Groq
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_client = Groq(api_key=groq_api_key)
        
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        answer = message.choices[0].message.content
    except Exception as e:
        print(f"  [API Error] {e}")
        answer = f"Error: {e}"
                
    latency  = time.time() - start
    tokens   = len(prompt.split()) + len(answer.split())

    return {
        "answer":               answer,
        "latency_sec":          latency,
        "tokens_approx":        tokens,
        "docs_retrieved":       len(subgraph["seed_papers"]),
        "related_docs":         len(subgraph["related_papers"]),
        "categories_traversed": len(subgraph["categories"]),
        "authors_found":        len(subgraph["authors"]),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline3(question=None, skip_ingest=True):
    """Run the full TigerGraph GraphRAG pipeline."""
    global _cached_token, _cached_conn, _cached_papers

    if question is None:
        question = QUESTION

    print("=" * 60)
    print("  Pipeline 3: TigerGraph GraphRAG")
    print("  Graph: Crystal | Vertex types: Papers, Authors, Categories")
    print("=" * 60)

    # Step 1 - Authenticate (cached)
    if _cached_conn is not None:
        print("\n[Step 1] Using cached TigerGraph connection ...")
        conn = _cached_conn
    else:
        print("\n[Step 1] Authenticating with TigerGraph Cloud ...")
        token = get_token()
        conn  = connect(token)
        _cached_conn = conn

        try:
            counts = conn.getVertexCount("*")
            print(f"  Connected! Vertex counts: {counts}")
        except Exception as e:
            _cached_conn = None
            raise SystemExit(f"[FATAL] Cannot query graph: {e}")

    # Step 2 - Ingest materials data (idempotent)
    if not skip_ingest:
        print("\n[Step 2] Ensuring materials-science data is in TigerGraph ...")
        ingest_materials(conn)
    else:
        print("\n[Step 2] Skipping ingestion to ensure high-speed querying ...")

    # Step 3 - Seed retrieval
    print(f"\n[Step 3] Retrieving seed papers from TigerGraph ...")
    print(f"  Question: {question}")
    t_start = time.time()
    seed_papers = find_seed_papers(conn, question, top_k=8)

    # Step 4 - Multi-hop traversal
    print(f"\n[Step 4] Multi-hop graph traversal ...")
    subgraph = graph_traversal(conn, seed_papers, question)
    retrieval_time = time.time() - t_start

    print(f"\n  Retrieval summary:")
    print(f"    Seed papers    : {len(subgraph['seed_papers'])}")
    print(f"    Categories     : {len(subgraph['categories'])}")
    print(f"    Related papers : {len(subgraph['related_papers'])}")
    print(f"    Authors        : {len(subgraph['authors'])}")
    print(f"    Retrieval time : {retrieval_time:.2f}s")

    # Step 5 - Generate answer
    print(f"\n[Step 5] Generating answer with Gemini ...")
    result = generate_answer(question, subgraph)
    result["retrieval_sec"] = retrieval_time

    print(f"\n  Tokens  : {result['tokens_approx']}")
    print(f"  Latency : {result['latency_sec']:.2f}s (generation) + {retrieval_time:.2f}s (retrieval)")
    print(f"  Docs    : {result['docs_retrieved']} seed + {result['related_docs']} related")
    print(f"  Answer  :\n{result['answer'][:600]}")

    print("\n" + "=" * 60)
    print("  Pipeline 3 complete.")
    print("=" * 60)

    return result


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else QUESTION
    run_pipeline3(q)