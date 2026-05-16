import os
import json
import time
import google.generativeai as genai

# Setup Gemini for generating reference answers
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAnqk1drgAlH519ZpNfHNZGNqoWvDPUM9M")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def load_questions(filepath):
    questions = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract just the question part
                q_text = line.split(". ", 1)[-1].strip()
                questions.append(q_text)
    return questions

def load_existing_ground_truth(filepath="ground_truth.json"):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["question"]: item["correct_answer"] for item in data}
    except Exception:
        return {}

def generate_ground_truth(questions, output_file="ground_truth.json"):
    existing_data = load_existing_ground_truth(output_file)
    print(f"Loaded {len(existing_data)} existing entries.")
    
    ground_truth = []
    
    for i, q in enumerate(questions):
        # Skip if we already have a valid answer
        if q in existing_data and existing_data[q] != "Error generating answer.":
            print(f"[{i+1}/{len(questions)}] Skipping (already exists): {q}")
            ground_truth.append({
                "question": q,
                "correct_answer": existing_data[q]
            })
            continue

        print(f"[{i+1}/{len(questions)}] Processing: {q}")
        prompt = (
            f"You are an expert materials scientist. Provide a highly accurate, concise, "
            f"and factual reference answer to the following question. "
            f"This answer will be used as the gold-standard ground truth to evaluate "
            f"other AI systems.\n\nQuestion: {q}\n\nCorrect Answer:"
        )
        
        success = False
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                answer = response.text.strip()
                if not answer:
                    raise Exception("Empty response")
                
                ground_truth.append({
                    "question": q,
                    "correct_answer": answer
                })
                print(f"  Successfully generated answer.")
                success = True
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(5)
        
        if not success:
            ground_truth.append({
                "question": q,
                "correct_answer": "Error generating answer."
            })
        
        # Save after every new generation to avoid data loss
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=4)
            
        time.sleep(10)  # Rate limiting for free tier
        
    print(f"\nGround truth successfully saved to {output_file}")

if __name__ == "__main__":
    if not os.path.exists("Equations.txt"):
        print("Error: Equations.txt not found.")
    else:
        questions = load_questions("Equations.txt")
        generate_ground_truth(questions)
