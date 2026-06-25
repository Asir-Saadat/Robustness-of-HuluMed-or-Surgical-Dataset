#!/usr/bin/env python3
"""
Grounded questions accuracy evaluation script.
Evaluates if Hulu-Med answers match original answers for visually grounded questions.
"""

import json
import os
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
    api_key=os.environ.get("OPENAI_API_KEY")
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


# Load the appropriate grounded answers file based on model size
json_file = f'/home/as5606/Datasets/Cholec_text_grounded_questions/hulu_med_{model_size}_grounded_answers.json'

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

for item in tqdm(hulumed_data, desc="Comparing grounded answers"):
    item_id = item['id']
    image = item['image']
    original_question = item['original_question']
    original_answer = item['original_answer']

    # Compare each grounded question answer with its expected answer
    grounded_qa_with_matches = []
    for grounded_qa in item['grounded_qa_with_hulu_answers']:
        question = grounded_qa['question']
        expected_answer = grounded_qa['original_answer']
        hulu_answer = grounded_qa['hulu_med_answer']

        # Compare expected answer with hulu answer
        match_result = compare_answers(expected_answer, hulu_answer)

        grounded_qa_with_matches.append({
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
        'grounded_qa_with_matches': grounded_qa_with_matches
    }
    results.append(result)

# Save to JSON FIRST (before any counting that might fail)
output_path = f'/home/as5606/Datasets/Cholec_text_grounded_questions/hulu_med_{model_size}_grounded_accuracy.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ Saved {len(results)} items to {output_path}")

# Count matches (after safe save)
try:
    total_grounded_qa = sum(len(item['grounded_qa_with_matches']) for item in results)
    match_count = sum(sum(1 for qa in item['grounded_qa_with_matches'] if qa['match'] == "MATCH") for item in results)
    accuracy = (match_count / total_grounded_qa) * 100

    print(f"\n{'='*50}")
    print(f"Model: Hulu-Med-{model_size}")
    print(f"Total grounded QA pairs: {total_grounded_qa}")
    print(f"Matched: {match_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")
except Exception as e:
    print(f"Warning: Could not calculate summary stats: {e}")
