#!/usr/bin/env python3
"""
Generate grounded visual questions for Cholec80 surgical VQA dataset.
Uses OpenAI API to answer visual questions about surgical tools.
"""

import json
import base64
import re
import os
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client (use environment variable for API key)
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=API_KEY)

QUESTION_BANK = [
    ("What colour is the {tool}?",
     ["silver/grey", "white", "black", "gold", "other"]),
    ("Is the {tool} on the left or right side of the image?",
     ["left", "right", "centre"]),
    ("Which direction is the tip of the {tool} pointing?",
     ["up", "down", "left", "right"]),
    ("How many surgical instruments are visible in the image?",
     ["1", "2", "3", "4+"]),
]

SYSTEM_PROMPT = """\
You are shown ONE laparoscopic cholecystectomy frame and told which surgical
tool is present in it. Answer SIMPLE, visually-grounded questions about that tool
- the kind anyone could answer by looking, with no medical knowledge.

For EACH question you are given an exact list of allowed answers. You MUST choose
one answer from that list and nothing else. If the relevant part of the tool is
not visible enough to judge, choose "not_visible" when it is offered; otherwise
choose the closest allowed answer.

Answer only from what is plainly visible in this single frame. Do not use surgical
knowledge, do not guess about anatomy, do not describe actions over time.

OUTPUT - return ONLY valid JSON, no preamble, no markdown fences:
{"answers": [{"question": "...", "answer": "..."}, ...]}
"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def answer_grounded(image_path: str, tool: str) -> list[dict]:
    """Answer the fixed question bank about `tool` for one frame."""
    # Build the question list with allowed answers spelled out for this tool.
    lines = []
    for q_tmpl, allowed in QUESTION_BANK:
        q = q_tmpl.format(tool=tool)
        lines.append(f'- "{q}"  allowed answers: {allowed}')
    bank_text = "\n".join(lines)

    user_text = (
        f"Tool present in this frame: {tool}\n\n"
        f"Answer each of these questions, one entry per question:\n{bank_text}"
    )

    b64 = _encode_image(image_path)

    response = client.responses.create(
        model="gpt-5.4-mini",
        instructions=SYSTEM_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{b64}"},
            ],
        }],
    )

    try:
        answers = _parse_json(response.output[0].content[0].text)["answers"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Parse failed for {image_path} ({tool}): {e}")
        return []

    # Keep only confident, in-set answers.
    allowed_by_q = {q_tmpl.format(tool=tool): set(a) for q_tmpl, a in QUESTION_BANK}
    kept = []
    for item in answers:
        q, a = item.get("question"), item.get("answer")
        if q in allowed_by_q and a in allowed_by_q[q]:
            kept.append(item)
    return kept


def main():
    """Main processing function."""
    print("Loading test dataset...")
    with open('/home/as5606/Datasets/cholec_formatted_data/cholec80_llava_test.json', 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    data_2 = []

    print("Processing samples...")
    for item in tqdm(data, desc="Processing"):
        answer = item['conversations'][1]['value'].lower()

        # Only get 'is used' (not 'is not used')
        if 'is used' in answer and 'is not used' not in answer:
            # Extract tool (word before "is")
            match = re.search(r'(\w+)\s+is', answer)
            if match:
                tool = match.group(1)

                try:
                    # Get grounded questions
                    result = answer_grounded(item['image'], tool=tool)

                    if result:
                        # Create new entry with original data + grounded questions
                        new_entry = {**item}
                        new_entry['grounded_qa_pairs'] = result
                        data_2.append(new_entry)

                except Exception as e:
                    print(f"Error processing {item['id']}: {e}")
            else:
                print(f"Tool not found in answer: {answer}")

    # Save output
    output_path = '/home/as5606/Datasets/Cholec_text_grounded_questions/cholec80_with_grounded_questions.json'
    print(f"\nSaving {len(data_2)} entries...")
    with open(output_path, 'w') as f:
        json.dump(data_2, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_path}")
    print(f"✓ Processed {len(data_2)} / {len(data)} samples")


if __name__ == "__main__":
    main()
