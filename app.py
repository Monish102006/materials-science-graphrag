import streamlit as st
import os
import json
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

# Import Pipeline 3 from run_pipeline3
from run_pipeline3 import run_pipeline3
from evaluate_accuracy import evaluate_single_answer, load_ground_truth
from generate_ground_truth import generate_ground_truth, load_questions
from huggingface_hub import InferenceClient
import evaluate

from dotenv import load_dotenv
load_dotenv()

# === UI SETUP ===
st.set_page_config(layout="wide", page_title="GraphRAG Benchmark Dashboard", page_icon="🐯")

# Setup configs
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)
model_name = 'llama-3.3-70b-versatile'

# Groq Pricing
PRICE_PER_TOKEN = 0.69 / 1_000_000  # avg $/token

# Try to connect to Vector DB
@st.cache_resource
def get_chroma_db():
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = chroma_client.get_collection(name="arxiv_papers", embedding_function=sentence_transformer_ef)
        return collection, True
    except Exception:
        return None, False

collection, db_available = get_chroma_db()

# --- Evaluation Resources ---
HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def get_eval_client():
    return InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)

@st.cache_resource
def get_bertscore():
    return evaluate.load("bertscore")

# Ground Truth Handling
if not os.path.exists("ground_truth.json"):
    st.warning("Ground truth file missing. Generating it now using Gemini...")
    questions_list = load_questions("Equations.txt")
    generate_ground_truth(questions_list)
    st.success("Ground truth generated!")

ground_truth_data = load_ground_truth()

def run_pipeline_1(question):
    start = time.time()
    message = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": f"Question: {question}\nAnswer based on your knowledge:"}],
        max_tokens=1024
    )
    latency = time.time() - start
    answer_text = message.choices[0].message.content
    tokens = len(question.split()) + len(answer_text.split())
    return {"Answer": answer_text, "Tokens": tokens, "Latency": round(latency, 2), "Cost": round(tokens * PRICE_PER_TOKEN, 6)}

def run_pipeline_2(question):
    if not db_available: return {"Answer": "ChromaDB not found.", "Tokens": 0, "Latency": 0, "Cost": 0}
    start = time.time()
    results = collection.query(query_texts=[question], n_results=5)
    context = "\n".join(results['documents'][0]) if results['documents'] else ""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    message = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    latency = time.time() - start
    answer_text = message.choices[0].message.content
    tokens = len(prompt.split()) + len(answer_text.split())
    return {"Answer": answer_text, "Tokens": tokens, "Latency": round(latency, 2), "Cost": round(tokens * PRICE_PER_TOKEN, 6)}

def run_pipeline_3_wrapper(question):
    """Run Pipeline 3: TigerGraph GraphRAG."""
    try:
        result = run_pipeline3(question)
        tokens = result["tokens_approx"]
        return {
            "Answer": result["answer"],
            "Tokens": tokens,
            "Latency": round(result["latency_sec"], 2),
            "Cost": round(tokens * PRICE_PER_TOKEN, 6)
        }
    except Exception as e:
        return {"Answer": f"GraphRAG Error: {e}", "Tokens": 0, "Latency": 0, "Cost": 0}

# === UI SETUP ===

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 4px;
    }
    .pipeline-header {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .p1-header { background: linear-gradient(90deg, #ff6b6b, #ee5a24); color: white; }
    .p2-header { background: linear-gradient(90deg, #4facfe, #00f2fe); color: white; }
    .p3-header { background: linear-gradient(90deg, #43e97b, #38f9d7); color: #1a1a2e; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">🐯 TigerGraph GraphRAG Benchmark Dashboard</h1>
    <p style="margin:0.3rem 0 0 0; opacity:0.9;">Proving GraphRAG reduces tokens while maintaining accuracy on Materials Science queries</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Benchmark", "📊 Aggregate Results", "🏗️ Architecture", "📋 Benchmark Report"])

# Load questions
questions = []
with open("Equations.txt", "r") as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            questions.append(line.split(". ", 1)[-1].strip())

# ================== TAB 1: Live Benchmark ==================
with tab1:
    st.subheader("Run a Live Comparison")
    st.markdown("Select a question and run it through all 3 pipelines simultaneously.")
    
    selected_q = st.selectbox("Select a benchmark question:", questions, key="live_q")
    
    if st.button("Run Benchmark 🚀", type="primary"):
        import concurrent.futures
        
        correct = ground_truth_data.get(selected_q, "")
        
        with st.spinner("Running all 3 pipelines concurrently..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                f1 = executor.submit(run_pipeline_1, selected_q)
                f2 = executor.submit(run_pipeline_2, selected_q)
                f3 = executor.submit(run_pipeline_3_wrapper, selected_q)
                
                res1 = f1.result()
                res2 = f2.result()
                res3 = f3.result()

        with st.spinner("Evaluating answers concurrently..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                e1 = executor.submit(evaluate_single_answer, get_eval_client(), get_bertscore(), selected_q, res1["Answer"], correct)
                e2 = executor.submit(evaluate_single_answer, get_eval_client(), get_bertscore(), selected_q, res2["Answer"], correct)
                e3 = executor.submit(evaluate_single_answer, get_eval_client(), get_bertscore(), selected_q, res3["Answer"], correct)
                
                eval1 = e1.result()
                eval2 = e2.result()
                eval3 = e3.result()

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="pipeline-header p1-header">Pipeline 1: Raw LLM</div>', unsafe_allow_html=True)
            st.metric("Tokens Used", res1["Tokens"])
            st.metric("Latency", f"{res1['Latency']}s")
            st.metric("Cost", f"${res1['Cost']}")
            st.metric("LLM Judge", "PASS ✅" if eval1["passed"] else "FAIL ❌")
            st.metric("BERTScore F1", f"{eval1['bertscore']:.4f}")
            with st.expander("Full Answer"):
                st.write(res1["Answer"])
            
        with col2:
            st.markdown('<div class="pipeline-header p2-header">Pipeline 2: Basic RAG</div>', unsafe_allow_html=True)
            st.metric("Tokens Used", res2["Tokens"])
            st.metric("Latency", f"{res2['Latency']}s")
            st.metric("Cost", f"${res2['Cost']}")
            st.metric("LLM Judge", "PASS ✅" if eval2["passed"] else "FAIL ❌")
            st.metric("BERTScore F1", f"{eval2['bertscore']:.4f}")
            with st.expander("Full Answer"):
                st.write(res2["Answer"])
            
        with col3:
            st.markdown('<div class="pipeline-header p3-header">Pipeline 3: GraphRAG</div>', unsafe_allow_html=True)
            st.metric("Tokens Used", res3["Tokens"])
            st.metric("Latency", f"{res3['Latency']}s")
            st.metric("Cost", f"${res3['Cost']}")
            st.metric("LLM Judge", "PASS ✅" if eval3["passed"] else "FAIL ❌")
            st.metric("BERTScore F1", f"{eval3['bertscore']:.4f}")
            with st.expander("Full Answer"):
                st.write(res3["Answer"])

        # Store eval results
        res1.update(eval1)
        res2.update(eval2)
        res3.update(eval3)

        # Save answers
        def save_answer(filename, title, question, response):
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"--- {title} ---\n")
                f.write(f"Question: {question}\n")
                f.write(f"Answer: {response['Answer']}\n")
                f.write(f"Tokens: {response['Tokens']}\n")
                f.write(f"Latency: {response['Latency']}s\n")
                f.write(f"LLM Judge: {'PASS' if response.get('passed') else 'FAIL'}\n")
                f.write(f"BERTScore: {response.get('bertscore', 0):.4f}\n\n")

        save_answer("normal llm.answers.txt", "Pipeline 1: Raw LLM", selected_q, res1)
        save_answer("ragllm_answers.txt", "Pipeline 2: Basic RAG", selected_q, res2)
        save_answer("tigergraph.answers.txt", "Pipeline 3: GraphRAG", selected_q, res3)
        
        # Token Reduction Highlight
        if res2["Tokens"] > 0 and res3["Tokens"] > 0:
            token_reduction = ((res2["Tokens"] - res3["Tokens"]) / res2["Tokens"]) * 100
            st.divider()
            
            hcol1, hcol2, hcol3 = st.columns(3)
            with hcol1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{token_reduction:.1f}%</div>
                    <div class="metric-label">Token Reduction (GraphRAG vs Basic RAG)</div>
                </div>""", unsafe_allow_html=True)
            with hcol2:
                cost_saving = ((res2["Cost"] - res3["Cost"]) / res2["Cost"]) * 100 if res2["Cost"] > 0 else 0
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">${res2['Cost'] - res3['Cost']:.6f}</div>
                    <div class="metric-label">Cost Savings per Query</div>
                </div>""", unsafe_allow_html=True)
            with hcol3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{eval3['bertscore']:.4f}</div>
                    <div class="metric-label">GraphRAG BERTScore F1</div>
                </div>""", unsafe_allow_html=True)

        # Charts
        st.divider()
        st.subheader("Visual Comparison")
        
        df = pd.DataFrame({
            "Pipeline": ["Raw LLM", "Basic RAG", "GraphRAG"],
            "Tokens": [res1["Tokens"], res2["Tokens"], res3["Tokens"]],
            "Latency (s)": [res1["Latency"], res2["Latency"], res3["Latency"]],
            "Cost ($)": [res1["Cost"], res2["Cost"], res3["Cost"]],
            "BERTScore": [eval1["bertscore"], eval2["bertscore"], eval3["bertscore"]]
        })
        
        colors = ['#ff6b6b', '#4facfe', '#43e97b']
        
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(data=[go.Bar(x=df["Pipeline"], y=df["Tokens"], marker_color=colors)])
            fig.update_layout(title="Tokens per Query (Lower is Better)", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(data=[go.Bar(x=df["Pipeline"], y=df["BERTScore"], marker_color=colors)])
            fig.update_layout(title="BERTScore F1 (Higher is Better)", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)


# ================== TAB 2: Aggregate Results ==================
with tab2:
    st.subheader("Aggregate Benchmark Results (All 30 Questions)")
    
    if os.path.exists("benchmark_report.json"):
        with open("benchmark_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)
        
        pipelines = report.get("pipelines", {})
        comp = report.get("comparison", {})
        
        # Headline metrics
        st.markdown("### Key Results")
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        with kcol1:
            val = comp.get("token_reduction_vs_basic_rag_pct", "N/A")
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}%</div>
                <div class="metric-label">Token Reduction vs Basic RAG</div>
            </div>""", unsafe_allow_html=True)
        with kcol2:
            val = comp.get("cost_reduction_vs_basic_rag_pct", "N/A")
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}%</div>
                <div class="metric-label">Cost Reduction vs Basic RAG</div>
            </div>""", unsafe_allow_html=True)
        with kcol3:
            p3 = pipelines.get("Pipeline 3: GraphRAG", {})
            val = p3.get("judge_pass_rate", "N/A")
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}%</div>
                <div class="metric-label">GraphRAG Judge Pass Rate</div>
            </div>""", unsafe_allow_html=True)
        with kcol4:
            val = p3.get("bertscore_f1_avg", "N/A")
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">GraphRAG BERTScore F1</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison table
        st.markdown("### Pipeline Comparison Table")
        rows = []
        metrics_to_show = [
            ("Questions Answered", "questions_answered"),
            ("Avg Tokens/Query", "avg_tokens"),
            ("Median Tokens/Query", "median_tokens"),
            ("Avg Latency (s)", "avg_latency_sec"),
            ("Cost per Query ($)", "cost_per_query_usd"),
            ("Cost per 1K Queries ($)", "cost_per_1000_queries_usd"),
            ("LLM Judge Pass Rate (%)", "judge_pass_rate"),
            ("BERTScore F1 (Avg)", "bertscore_f1_avg"),
        ]
        
        for label, key in metrics_to_show:
            row = {"Metric": label}
            for pname in ["Pipeline 1: Raw LLM", "Pipeline 2: Basic RAG", "Pipeline 3: GraphRAG"]:
                short_name = pname.split(": ")[1]
                row[short_name] = pipelines.get(pname, {}).get(key, "N/A")
            rows.append(row)
        
        df_table = pd.DataFrame(rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        
        # Charts
        st.markdown("### Visual Comparisons")
        pipeline_names = ["Raw LLM", "Basic RAG", "GraphRAG"]
        colors = ['#ff6b6b', '#4facfe', '#43e97b']
        
        c1, c2 = st.columns(2)
        with c1:
            avg_tokens = [pipelines.get(f"Pipeline {i}: {n}", {}).get("avg_tokens", 0) 
                         for i, n in enumerate(pipeline_names, 1)]
            fig = go.Figure(data=[go.Bar(x=pipeline_names, y=avg_tokens, marker_color=colors)])
            fig.update_layout(title="Average Tokens per Query", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            pass_rates = [pipelines.get(f"Pipeline {i}: {n}", {}).get("judge_pass_rate", 0) or 0
                         for i, n in enumerate(pipeline_names, 1)]
            fig = go.Figure(data=[go.Bar(x=pipeline_names, y=pass_rates, marker_color=colors)])
            fig.update_layout(title="LLM Judge Pass Rate (%)", template="plotly_dark", height=350, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        
        c3, c4 = st.columns(2)
        with c3:
            costs = [pipelines.get(f"Pipeline {i}: {n}", {}).get("cost_per_1000_queries_usd", 0) or 0
                    for i, n in enumerate(pipeline_names, 1)]
            fig = go.Figure(data=[go.Bar(x=pipeline_names, y=costs, marker_color=colors)])
            fig.update_layout(title="Cost per 1,000 Queries ($)", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with c4:
            bert_scores = [pipelines.get(f"Pipeline {i}: {n}", {}).get("bertscore_f1_avg", 0) or 0
                          for i, n in enumerate(pipeline_names, 1)]
            fig = go.Figure(data=[go.Bar(x=pipeline_names, y=bert_scores, marker_color=colors)])
            fig.update_layout(title="BERTScore F1 Average", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Benchmark report not generated yet. Run `python benchmark_report.py` first.")


# ================== TAB 3: Architecture ==================
with tab3:
    st.subheader("System Architecture")
    st.markdown("""
    This project benchmarks three AI pipelines on **Materials Science** queries using a dataset 
    of **10,000 arXiv papers** (~2.7M tokens). The architecture demonstrates how GraphRAG's 
    multi-hop graph traversal achieves significant token reduction compared to traditional vector RAG.
    """)
    
    if os.path.exists("architecture_diagram.png"):
        st.image("architecture_diagram.png", use_column_width=True)
    else:
        st.info("Architecture diagram not found. Place `architecture_diagram.png` in the project root.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### Pipeline 1: Raw LLM
        - **No retrieval** - pure LLM inference
        - Groq LLaMA 3.3 70B
        - Baseline for comparison
        """)
    with col2:
        st.markdown("""
        #### Pipeline 2: Basic RAG
        - ChromaDB vector store
        - all-MiniLM-L6-v2 embeddings
        - Top-5 similarity search
        """)
    with col3:
        st.markdown("""
        #### Pipeline 3: GraphRAG
        - TigerGraph Cloud (Savanna)
        - Multi-hop traversal
        - Papers -> Categories -> Related Papers -> Authors
        """)


# ================== TAB 4: Benchmark Report ==================
with tab4:
    st.subheader("Full Benchmark Report")
    
    if os.path.exists("benchmark_report.md"):
        with open("benchmark_report.md", "r", encoding="utf-8") as f:
            report_md = f.read()
        st.markdown(report_md)
    else:
        st.warning("Run `python benchmark_report.py` to generate the benchmark report.")
    
    st.markdown("---")
    st.markdown("### Download Report")
    
    if os.path.exists("benchmark_report.json"):
        with open("benchmark_report.json", "r", encoding="utf-8") as f:
            report_json = f.read()
        st.download_button("Download JSON Report", report_json, "benchmark_report.json", "application/json")
    
    if os.path.exists("benchmark_report.md"):
        with open("benchmark_report.md", "r", encoding="utf-8") as f:
            report_md = f.read()
        st.download_button("Download Markdown Report", report_md, "benchmark_report.md", "text/markdown")
