import time
import re

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

text = "sample text " * 200
papers = [{"text": text} for _ in range(10000)]
q_tokens = {"sample"}

start = time.time()
scored = []
for p in papers:
    p_tokens = tokenize(p["text"])
    overlap = len(q_tokens & p_tokens)
    if overlap >= 2:
        scored.append(overlap)
print(f"Time: {time.time() - start}s")
