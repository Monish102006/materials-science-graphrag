import os
import json
import re
import evaluate
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini for Judging
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
judge_model = genai.GenerativeModel('gemini-2.5-flash')

# BERTScore will be loaded dynamically when needed
bertscore = None

JUDGE_PROMPT = """Grade the system's answer.
Question: {q}
Correct answer: {correct}
System answer: {answer}
PASS = the system answer correctly addresses the question with no major errors.
FAIL = the answer is wrong, missing, or contradicts the correct answer."""

API_KEYS = [
    os.getenv("GOOGLE_API_KEY")
]
current_key_idx = 0

def get_judge_verdict(prompt):
    global current_key_idx
    max_retries = len(API_KEYS)
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=API_KEYS[current_key_idx])
            judge_model = genai.GenerativeModel('gemini-2.5-flash')
            response = judge_model.generate_content(prompt)
            content = response.text.upper()
            return "PASS" in content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                print(f"  [Judge Rate Limit] Key {current_key_idx} failed. Switching... (Attempt {attempt+1}/{max_retries})")
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                time.sleep(1)
            else:
                print(f"  Judge error: {e}")
                return False
    return False

def evaluate_single_answer(eval_client, bertscore_metric, question, answer, correct):
    """
    Evaluates a single answer using both LLM-as-a-judge and BERTScore.
    Used by the Streamlit dashboard for real-time evaluation.
    """
    if not correct:
        return {"passed": False, "bertscore": 0.0}

    # 1. LLM-as-a-Judge
    prompt = JUDGE_PROMPT.format(q=question, correct=correct, answer=answer)
    passed = False
    try:
        # InferenceClient.text_generation
        response = eval_client.text_generation(prompt, max_new_tokens=10)
        passed = "PASS" in response.upper()
    except Exception as e:
        # Fallback to Gemini if HF client fails (optional, but let's try to be robust)
        try:
            verdict = get_judge_verdict(prompt)
            passed = verdict if verdict is not None else False
        except:
            passed = False
    
    # 2. BERTScore
    try:
        results = bertscore_metric.compute(
            predictions=[answer],
            references=[correct],
            lang="en",
            rescale_with_baseline=False
        )
        score = results["f1"][0]
    except Exception as e:
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
