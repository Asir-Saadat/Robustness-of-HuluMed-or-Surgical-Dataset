#!/usr/bin/env python3
"""
Hallucinatory questions accuracy evaluation script.
Evaluates if Hulu-Med correctly identifies unanswerable questions.
"""

import json
import sys
from tqdm import tqdm
from openai import OpenAI

# Get model size from command line argument
if len(sys.argv) < 2:
    print("Usage: python3 test_accuracy.py <model_size>")
    print("Example: python3 test_accuracy.py 4B")
    sys.exit(1)

model_size = sys.argv[1]
print(f"Running evaluation for Hulu-Med-{model_size}")

# Initialize OpenAI client
client = OpenAI(
    api_key="os.environ.get("OPENAI_API_KEY")"
)

# System prompt for answer comparison
SYSTEM_PROMPT = """\
You are an answer-matching evaluator.

Your task is to compare two answers and decide whether they have the same meaning.

The answers do not need to use the exact same words. They should be considered a match if they express the same core idea, refer to the same entity, or would be accepted as semantically equivalent in context.

You should focus on meaning, not wording.

Return only one of the following labels:

MATCH
NO_MATCH

Guidelines:
- Mark MATCH if both answers convey the same meaning, even with different wording.
- Mark MATCH if one answer is more detailed but does not contradict the other.
- Mark MATCH for synonyms, paraphrases, abbreviations, or equivalent medical/scientific terms.
- Mark NO_MATCH if the answers refer to different entities, objects, actions, locations, numbers, or concepts.
- Mark NO_MATCH if one answer contradicts the other.
- Mark NO_MATCH if one answer is too vague to confirm equivalence.
- Ignore capitalization, punctuation, grammar mistakes, and minor wording differences.

Input format:
Answer 1: <first answer>
Answer 2: <second answer>

Output format:
MATCH or NO_MATCH
"""


def compare_answers(answer1: str, answer2: str) -> str:
    """Compare two answers and return whether they match in meaning.

    Returns: "MATCH" or "NO_MATCH"
    """

    user_text = (
        f"Answer 1: {answer1}\n"
        f"Answer 2: {answer2}"
    )

    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            instructions=SYSTEM_PROMPT,
            input=user_text,
        )

        result = response.output_text.strip()

        # Extract MATCH or NO_MATCH from response
        if "MATCH" in result.upper():
            if "NO_MATCH" in result.upper():
                no_match_idx = result.upper().find("NO_MATCH")
                match_idx = result.upper().find("MATCH")
                if no_match_idx < match_idx:
                    return "NO_MATCH"
            return "MATCH"
        else:
            return "NO_MATCH"

    except Exception as e:
        print(f"Error comparing answers: {e}")
        return None


# Load the appropriate hallucinatory answers file based on model size
json_file = f'/home/as5606/Datasets/Cholec_text_hallucinatory_questions/hulu_med_{model_size}_hallucinatory_answers.json'

print(f"Loading {json_file}...")
try:
    with open(json_file, 'r') as f:
        hulumed_data = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found: {json_file}")
    sys.exit(1)

print(f"Loaded {len(hulumed_data)} samples")

# Compare answers
results = []

for item in tqdm(hulumed_data, desc="Comparing hallucinatory answers"):
    item_id = item['id']
    image = item['image']
    original_question = item['original_question']
    original_answer = item['original_answer']

    # Compare each new question with hulu answers
    new_questions_with_matches = []
    for new_qa in item['new_questions_with_hulu_answers']:
        question_type = new_qa['type']
        question = new_qa['question']
        hulu_answer = new_qa['hulu_med_answer']

        # For hallucinatory questions, check if model correctly says "I don't know"
        expected_answer = "I don't know"

        # Compare answers
        match_result = compare_answers(expected_answer, hulu_answer)

        new_questions_with_matches.append({
            'type': question_type,
            'question': question,
            'expected_answer': expected_answer,
            'hulu_med_answer': hulu_answer,
            'match': match_result
        })

    result = {
        'id': item_id,
        'image': image,
        'original_question': original_question,
        'original_answer': original_answer,
        'new_questions_with_matches': new_questions_with_matches
    }
    results.append(result)

# Save to JSON FIRST (before any counting that might fail)
output_path = f'/home/as5606/Datasets/Cholec_text_hallucinatory_questions/hulu_med_{model_size}_hallucinatory_accuracy.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ Saved {len(results)} items to {output_path}")

# Count matches (after safe save)
try:
    total_new_qa = sum(len(item['new_questions_with_matches']) for item in results)
    match_count = sum(sum(1 for qa in item['new_questions_with_matches'] if qa['match'] == "MATCH") for item in results)
    accuracy = (match_count / total_new_qa) * 100

    print(f"\n{'='*50}")
    print(f"Model: Hulu-Med-{model_size}")
    print(f"Total hallucinatory QA pairs: {total_new_qa}")
    print(f"Correctly said 'I don't know': {match_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")
except Exception as e:
    print(f"Warning: Could not calculate summary stats: {e}")
