import os
import json
import re
import evaluate
import time
import google.generativeai as genai
from dotenv import load_dotenv
import threading

load_dotenv()

bert_lock = threading.Lock()

# Initialize Gemini for Judging
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
judge_model = genai.GenerativeModel('gemini-1.5-flash')

# BERTScore will be loaded dynamically when needed
bertscore = None

JUDGE_PROMPT = """You are an expert Materials Science judge. Grade the 'System Answer' against the 'Correct Answer'.
Your output MUST be either 'PASS' or 'FAIL'. No other text.

Criteria:
- PASS: The system answer is factually correct and matches the core meaning of the correct answer.
- FAIL: The system answer is wrong, contains hallucinations, or misses the primary point.

Question: {q}
Correct Answer: {correct}
System Answer: {answer}

Grade (PASS/FAIL):"""

API_KEYS = [
    os.getenv("GOOGLE_API_KEY")
]
current_key_idx = 0

def get_judge_verdict(prompt):
    """Fallback judge using Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return "PASS" in response.text.upper()
    except Exception as e:
        print(f"  Gemini Judge error: {e}")
        return False

def evaluate_single_answer(eval_client, bertscore_metric, question, answer, correct):
    """
    Evaluates a single answer using both LLM-as-a-judge and BERTScore.
    Used by the Streamlit dashboard for real-time evaluation.
    """
    if not correct:
        return {"passed": False, "bertscore": 0.0}

    # 1. LLM-as-a-Judge
    passed = False
    try:
        # Use Chat Completion for better instruction following on Llama 3.1
        messages = [{"role": "user", "content": JUDGE_PROMPT.format(q=question, correct=correct, answer=answer)}]
        response = eval_client.chat_completion(messages=messages, max_tokens=10)
        verdict = response.choices[0].message.content.upper()
        passed = "PASS" in verdict
    except Exception as e:
        print(f"HF Judge Error: {e}. Falling back to Gemini...")
        passed = get_judge_verdict(JUDGE_PROMPT.format(q=question, correct=correct, answer=answer))
    
    # 2. BERTScore
    try:
        with bert_lock:
            results = bertscore_metric.compute(
                predictions=[answer],
                references=[correct],
                lang="en",
                model_type="distilbert-base-uncased",
                rescale_with_baseline=False
            )
            score = results["f1"][0]
    except Exception as e:
        print(f"BERTScore Error: {e}")
        score = 0.0
        
    return {"passed": passed, "bertscore": score}

def load_ground_truth(filepath="ground_truth.json"):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run generate_ground_truth.py first.")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["question"]: item["correct_answer"] for item in data}

def parse_answers_file(filepath):
    """Parses the generated answer files from the benchmark run."""
    answers = {}
    if not os.path.exists(filepath):
        return answers
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by the title separator
    blocks = re.split(r'--- .*? ---\n', content)
    for block in blocks:
        if not block.strip():
            continue
        
        # Extract Question and Answer
        # Adjusted regex to handle multi-line answers better
        q_match = re.search(r'Question: (.*?)\nAnswer: (.*?)\nTokens:', block, re.DOTALL)
        if q_match:
            q = q_match.group(1).strip()
            a = q_match.group(2).strip()
            answers[q] = a
            
    return answers

def evaluate_pipeline(pipeline_name, answers_dict, ground_truth):
    print(f"\nEvaluating {pipeline_name}...")
    pipeline_outputs = []
    truths = []
    
    # Match answers with ground truth
    for q, correct in ground_truth.items():
        if q in answers_dict:
            pipeline_outputs.append(answers_dict[q])
            truths.append({"question": q, "correct_answer": correct})
    
    if not pipeline_outputs:
        print(f"  No matching answers found for {pipeline_name}.")
        return None
        
    print(f"  Found {len(pipeline_outputs)} answers to evaluate.")
    
    judge_results = []
    use_judge = True
    for i, (output, truth) in enumerate(zip(pipeline_outputs, truths)):
        if use_judge:
            print(f"  [{i+1}/{len(pipeline_outputs)}] Judging answer...")
            prompt = JUDGE_PROMPT.format(
                q=truth["question"],
                correct=truth["correct_answer"],
                answer=output
            )
            verdict = get_judge_verdict(prompt)
            if verdict is None: # Special case for quota error
                print("  Quota reached for Judge. Skipping remaining judge calls for this pipeline.")
                use_judge = False
            else:
                judge_results.append(verdict)
            time.sleep(1)
            
    print(f"  Calculating BERTScore for {len(pipeline_outputs)} samples...")
    try:
        global bertscore
        if bertscore is None:
            bertscore = evaluate.load("bertscore")
        bert_results = bertscore.compute(
            predictions=pipeline_outputs,
            references=[t["correct_answer"] for t in truths],
            lang="en",
            model_type="distilbert-base-uncased",
            rescale_with_baseline=False
        )
        bertscore_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])
    except Exception as e:
        print(f"  Error calculating BERTScore: {e}")
        bertscore_f1 = 0.0
        
    pass_rate = sum(judge_results) / len(judge_results) if judge_results else 0.0
    
    print(f"  Result -> LLM Judge Pass Rate: {pass_rate*100:.1f}%, BERTScore F1 (Rescaled): {bertscore_f1:.4f}")
    return {
        "llm_judge_pass_rate": pass_rate,
        "bertscore_f1": bertscore_f1
    }

if __name__ == "__main__":
    ground_truth = load_ground_truth()
    if not ground_truth:
        exit(1)
        
    print("Parsing pipeline answer files...")
    pipeline_1_answers = parse_answers_file("normal llm.answers.txt")
    pipeline_2_answers = parse_answers_file("ragllm_answers.txt")
    pipeline_3_answers = parse_answers_file("tigergraph.answers.txt")
    
    results = {}
    
    if pipeline_1_answers:
        results["Pipeline 1: Raw LLM"] = evaluate_pipeline("Pipeline 1: Raw LLM", pipeline_1_answers, ground_truth)
    if pipeline_2_answers:
        results["Pipeline 2: Basic RAG"] = evaluate_pipeline("Pipeline 2: Basic RAG", pipeline_2_answers, ground_truth)
    if pipeline_3_answers:
        results["Pipeline 3: GraphRAG"] = evaluate_pipeline("Pipeline 3: GraphRAG", pipeline_3_answers, ground_truth)
        
    print("\n" + "="*50)
    print("FINAL ACCURACY METRICS")
    print("="*50)
    for name, res in results.items():
        if res:
            print(f"{name}:")
            print(f"  LLM-as-a-Judge Pass Rate : {res['llm_judge_pass_rate']*100:.1f}% (Target: >= 90%)")
            print(f"  BERTScore F1 (Rescaled)  : {res['bertscore_f1']:.4f} (Target: >= 0.55)")
    print("="*50)
