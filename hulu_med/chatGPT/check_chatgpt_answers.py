#!/usr/bin/env python3
"""
ChatGPT accuracy evaluation for "how many" questions only.
Evaluates if ChatGPT answers match original answers for counting questions.
"""

import json
import os
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="os.environ.get("OPENAI_API_KEY")")


# System prompt for "how many" questions
SYSTEM_PROMPT = """\
You are an answer-matching evaluator for surgical VQA counting questions.

Your task is to compare two answers about quantity and decide whether they express the same number.

Return only one of the following labels:

MATCH
NO_MATCH

Guidelines:
- Mark MATCH if both answers refer to the SAME QUANTITY, regardless of wording
- Mark MATCH for number variations: "2" matches "Two" or "Two tools are operating"
- Mark MATCH for variations: "1" matches "One" or "One tool" or "There is one tool"
- Mark MATCH for written forms: "3" matches "three" or "There are three tools"
- Mark MATCH even if one answer is more verbose: "2" matches "Two surgical instruments are visible"
- Mark MATCH for number words: "four" matches "4" matches "There are 4 tools operating"
- Mark NO_MATCH if the numbers differ: "2" vs "3"
- Mark NO_MATCH if one says "no tools" or "0" and the other says there are tools
- Ignore capitalization, punctuation, and extra wording

CRITICAL EXAMPLES:
Answer 1: "2" | Answer 2: "Two tools are operating." → MATCH (both say 2)
Answer 1: "1" | Answer 2: "One tool" → MATCH (both say 1)
Answer 1: "3" | Answer 2: "There are three surgical instruments visible." → MATCH (both say 3)
Answer 1: "0" | Answer 2: "No tools are visible." → MATCH (both say 0)
Answer 1: "2" | Answer 2: "Three tools" → NO_MATCH (2 vs 3)
Answer 1: "1" | Answer 2: "Two tools" → NO_MATCH (1 vs 2)

Input format:
Answer 1: <first answer>
Answer 2: <second answer>

Output format:
MATCH or NO_MATCH
"""

def compare_answers(answer1: str, answer2: str) -> str:
    """Compare two answers and return whether they match in meaning."""
    user_text = f"Answer 1: {answer1}\nAnswer 2: {answer2}"
    
    try:
        response = client.responses.create(
            model="gpt-5.4-nano",
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

# Load ChatGPT answers
json_file = '/home/as5606/projects/Hulu-Med/chatGPT/chatgpt_answers.json'

print(f"Loading {json_file}...")
try:
    with open(json_file, 'r') as f:
        chatgpt_data = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found: {json_file}")
    exit(1)

print(f"Loaded {len(chatgpt_data)} samples\n")

# Add match field for "how many" questions only
how_many_count = 0
matched_count = 0

for item in tqdm(chatgpt_data, desc="Adding match field for 'how many' questions"):
    question = item['question'].lower()

    # Only evaluate "how many" questions
    if "how many" in question:
        how_many_count += 1
        original_answer = item['original_answer']
        chatgpt_answer = item['chatgpt_answer']

        # Compare answers
        match = compare_answers(original_answer, chatgpt_answer)
        item['match'] = match

        if match == 'MATCH':
            matched_count += 1

# Overwrite original file with updated data
with open(json_file, 'w') as f:
    json.dump(chatgpt_data, f, indent=2, ensure_ascii=False)

print(f"✓ Updated and saved {len(chatgpt_data)} items to {json_file}")

# Print summary
print(f"\n{'='*60}")
print(f"ChatGPT 'HOW MANY' QUESTIONS EVALUATION")
print(f"{'='*60}")
print(f"Total 'how many' questions: {how_many_count}")
print(f"Matched: {matched_count}")
if how_many_count > 0:
    accuracy = (matched_count / how_many_count * 100)
    print(f"Accuracy: {accuracy:.2f}%")
print(f"{'='*60}")
