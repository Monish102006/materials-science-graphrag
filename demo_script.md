# GraphRAG Inference Hackathon: Materials Science Discovery
## Demo Video Script (Voice-Over Only)

**Target Length:** 2-3 minutes
**Tone:** Professional, engaging, and technical.
**Requirements:** Screen recording software (e.g., OBS, Loom), Streamlit app running locally, architecture diagram open.

---

### Scene 1: Introduction (0:00 - 0:25)

**Visual:** 
Start with your browser open to the **GraphRAG Inference Hackathon Notion page**. Briefly scroll to the project description or requirements section, then switch to your project's GitHub repository or title slide. Finally, transition to the architecture diagram (`architecture_diagram.png`).

**Audio (Speaker):**
"Hi everyone! We're excited to present our project for the **TigerGraph GraphRAG Inference Hackathon**. 

Looking at the challenge set out here on Notion, we wanted to tackle the core problem: how do we make RAG more accurate and efficient for complex research? 

We chose the **Materials Science dataset** specifically because of its extreme complexity. In this field, breakthrough discovery relies on understanding deep, multi-relational connections between materials, properties, and researchers—data that standard search often misses.

Our project proves that graph-structured retrieval completely outperforms standard vector RAG. Let me show you how."

---

### Scene 2: Architecture & Setup (0:20 - 0:45)

**Visual:** 
Keep the architecture diagram on screen. Use your mouse cursor (with highlighting) to point to the 3 pipelines as you mention them.

**Audio (Speaker):**
"We processed 10,000 arXiv papers on condensed matter physics. To really benchmark performance, we built three parallel pipelines:

First, a **Raw LLM** using Groq's LLaMA 3.3 70B—no retrieval, just raw knowledge.
Second, a **Basic RAG** using ChromaDB for vector similarity search.
And third, our **GraphRAG pipeline** powered by TigerGraph Cloud. 

For GraphRAG, we do a multi-hop traversal: starting from keyword matches, moving to papers, expanding to categories, and grabbing related papers and authors to build a highly focused context subgraph."

---

### Scene 3: The Dashboard Demo (0:45 - 1:45)

**Visual:** 
Switch to the live Streamlit dashboard (`streamlit run app.py`). 
Show the UI, then type a sample multi-hop or relational question from your dataset (e.g., "What are the common authors or categories between paper X and paper Y?"). Hit run.

**Audio (Speaker):**
"Let's see it in action. Here is our interactive Streamlit dashboard. 

You'll notice we've provided a **dropdown of 30 curated benchmark questions**. We did this to ensure a standardized, rigorous evaluation. By using a fixed set of questions across all three pipelines, we can directly compare performance and accuracy against our expert-generated ground truth.

I'm going to select a complex multi-hop question. While this runs, all three pipelines are processing the same query simultaneously."

*(Wait a moment for results to appear)*

And here are the results side-by-side. 
As you can see, the **Raw LLM** gave a generic answer and hallucinated a few details. 
The **Basic RAG** pulled in a massive amount of text—over 1,000 tokens—but still missed the relational context. 
But look at the **GraphRAG** output. It used fewer tokens than Basic RAG, but delivered a highly accurate, citation-backed answer by traversing the connections in TigerGraph."

---

### Scene 4: Metrics and Evaluation (1:45 - 2:30)

**Visual:** 
Scroll down in the Streamlit app to the Benchmark Report / Metrics section. Highlight the LLM-as-a-Judge and BERTScore charts.

**Audio (Speaker):**
"But we didn't just eyeball it; we built an automated accuracy evaluation framework using HuggingFace's LLM-as-a-judge and BERTScore against expert-generated ground truth.

The numbers speak for themselves. 
GraphRAG achieved a **55.6% LLM Judge Pass rate**—which is double the accuracy of Basic RAG. It also scored significantly higher on BERTScore.

And because GraphRAG provides a focused, precise context window, it used **19% fewer tokens** than Basic RAG, making it cheaper and highly efficient at just 59 cents per 1,000 queries."

---

### Scene 5: Conclusion (2:30 - 2:45)

**Visual:** 
Switch back to the GitHub Repo or a final summary slide with your contact info or social media links.

**Audio (Speaker):**
"By integrating TigerGraph, we eliminated the noise of standard vector search and allowed the LLM to 'see' the actual relationships in the data. 

All our code, benchmarks, and the detailed blog post are available on GitHub. Thank you to TigerGraph for hosting this hackathon, and thanks for watching!"

---

### Filming Tips:
- **Clean Desktop:** Hide icons and use a neutral wallpaper for a professional look.
- **Microphone Quality:** Use a decent mic to ensure your voice is clear, as there's no face to focus on.
- **Rehearse:** Run through the Streamlit query a few times before recording to ensure it loads smoothly without unexpected errors.
- **Pre-load (Optional):** If the APIs are taking too long on live video, you can pre-load a query and explain the results, or edit out the waiting time.
- **Cursor Focus:** Use a tool to highlight your mouse cursor so viewers can follow where you are pointing on the dashboard and architecture diagram.
