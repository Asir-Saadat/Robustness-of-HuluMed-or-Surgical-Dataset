#!/usr/bin/env python3
"""
Generate unanswerable questions for Cholec80 surgical VQA dataset.
Uses OpenAI API to create synthetic questions from verified facts.
"""

import json
import re
import os
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client (use environment variable for API key)
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=API_KEY)

# System prompts
SYSTEM_PROMPT = """\
You generate UNANSWERABLE questions for a surgical visual-question-answering
benchmark built on laparoscopic cholecystectomy (Cholec80) video frames.

Each input describes one VERIFIED fact about a single frame: a surgical TOOL,
a surgical PHASE, and a LABEL ("yes" or "no") saying whether that tool is used
in that phase in this frame. Turn this verified fact into questions that have
NO determinate answer from a single frame, so the correct model response is to
abstain ("I don't know" / "cannot be determined").

CRITICAL DISTINCTION: A question whose answer is "no" or "zero" is ANSWERABLE,
not unanswerable. "Is the scissors used here?" answered "no" is answerable.
An unanswerable question is one where no determinate answer exists. NEVER
produce a question that merely has a negative answer.

Question types:

1. FALSE PREMISE - ONLY when LABEL = "no". The tool is NOT used in this phase,
   so any question presupposing it IS used has no answer. Ask what it is doing,
   where it is, why it is used, or what it interacts with - wrongly assuming
   its presence.
   Input: tool="specimen bag", phase="calot triangle dissection", label="no"
   - "What is the specimen bag retrieving during calot triangle dissection?"

2. NON-PERCEIVABLE ATTRIBUTE - ONLY when LABEL = "yes". The tool IS used, but
   ask a property a single image cannot reveal: elapsed time, applied force,
   temperature, manufacturer, intent, number of past uses. AVOID anything
   visually inferable (open/closed, left/right, currently-visible count).
   Input: tool="hook", phase="calot triangle dissection", label="yes"
   - "How much force is the hook applying during calot triangle dissection?"

3. TEMPORAL / FUTURE - any label. Ask about events after this frame that the
   frame cannot determine.
   - "Which phase will the surgeon begin next?"

4. EXTERNAL KNOWLEDGE - any label. Ask about information not present in any
   surgical frame.
   - "What is the patient's age?"

RULES:
- TYPE DIVERSITY: Every question you return MUST be a DIFFERENT type. Never
  return two questions of the same type. If asked for 2 questions, they must
  come from 2 distinct types above.
- Respect the label: use type 1 ONLY for label "no"; use type 2 ONLY for
  label "yes". Types 3 and 4 are allowed for either label.
- Each question must be fluent and natural, as a surgeon might phrase it.
- Do NOT ask yes/no questions (they tend to be answerable).
- Do NOT restate whether the tool is present or used.
- Keep every question about this single frame only.

OUTPUT - return ONLY valid JSON, no preamble, no markdown fences:
{"questions": [{"type": "false_premise", "question": "..."}, ...]}
"""

COUNT_PROMPT = """\
You are given a counting question about a single Cholec80 surgical frame and
its VERIFIED integer answer N (the number of tools operating in this frame).
Generate UNANSWERABLE questions — ones with no determinate answer from this
single frame, where the correct response is to abstain.

CRITICAL: A count answered "0" or any small number is ANSWERABLE, not
unanswerable. Do NOT just produce easier counting questions.

Types:
1. OUT-OF-RANGE PREMISE: presuppose MORE than N tools, so the question refers
   to a tool that does not exist.
   N=2 -> "What is the third tool doing?" / "How are the four tools coordinated?"
2. COMPLEXITY: ask for an exact count that a single frame cannot give —
   requiring procedure history or too fine to count.
   - "How many times has each tool been inserted so far in the procedure?"
3. TEMPORAL: ask about counts in the future.
   - "How many tools will be operating in the next phase?"

RULES:
- Every question must be fluent and natural for a surgeon.
- Do NOT ask yes/no questions.
- For type 1, the presupposed number MUST exceed N.

OUTPUT — return ONLY valid JSON, no markdown:
{"questions": [{"type": "out_of_range", "question": "..."}, ...]}
"""


def _parse_json(raw: str) -> dict:
    """Strip accidental code fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def extract_from_answer(answer: str) -> dict:
    """
    Extract tool, phase, and label from answer.
    Example: 'scissors is not used in preparation' ->
    {'tool': 'scissors', 'label': 'no', 'phase': 'preparation'}
    """
    answer_lower = answer.lower()

    # Extract tool (word before "is")
    tool_match = re.search(r'(\w+)\s+is', answer_lower)
    tool = tool_match.group(1) if tool_match else None

    # Extract label (yes if "is used", no if "is not used")
    label = 'no' if 'is not used' in answer_lower else 'yes'

    # Extract phase (everything after "in")
    phase_match = re.search(r'in\s+(.+?)(?:\.|$)', answer_lower)
    phase = phase_match.group(1).strip() if phase_match else None

    return {
        'tool': tool,
        'label': label,
        'phase': phase
    }


def make_unanswerable(tool: str, phase: str, label: str, n: int = 2) -> list:
    """Generate `n` unanswerable questions for one verified (tool, phase, label) item."""
    label = label.strip().lower()
    assert label in {"yes", "no"}, "label must be 'yes' or 'no'"

    user_msg = (
        f"tool: {tool}\n"
        f"phase: {phase}\n"
        f"label: {label}\n"
        f"Generate {n} unanswerable questions appropriate to this label."
    )

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=user_msg,
        instructions=SYSTEM_PROMPT,
    )

    try:
        output_text = response.output[0].content[0].text
        return _parse_json(output_text)["questions"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Parse failed for ({tool}, {phase}, {label}): {e}")
        return []


def make_unanswerable_count(question: str, count: int, n: int = 2) -> list:
    """Generate `n` unanswerable questions for a counting question."""
    user_msg = (
        f"counting question: {question}\n"
        f"verified answer N: {count}\n"
        f"Generate {n} unanswerable questions."
    )

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=user_msg,
        instructions=COUNT_PROMPT,
    )

    try:
        output_text = response.output[0].content[0].text
        return _parse_json(output_text)["questions"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Parse failed for ({question}, N={count}): {e}")
        return []


def main():
    """Main processing function."""
    print("Loading test dataset...")
    with open('/home/as5606/Datasets/cholec_formatted_data/cholec80_llava_test.json', 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    data_2 = []

    print("Processing samples...")
    for info in tqdm(data, desc="Processing"):
        question = info['conversations'][0]['value']
        answer = info['conversations'][1]['value']

        new_entry = {**info}  # Copy original entry

        try:
            if question == "how many tools are operating?\n<image>":
                new_questions = make_unanswerable_count(question, int(answer))
            else:
                # Clean question for extraction
                clean_question = question.replace('\n<image>', '').strip()

                # Only process yes/no type answers
                if 'is used' in answer.lower() or 'is not used' in answer.lower():
                    results = extract_from_answer(answer)
                    new_questions = make_unanswerable(
                        results['tool'],
                        results['phase'],
                        results['label']
                    )
                else:
                    continue

            # Add new questions to entry
            new_entry['new_questions'] = new_questions
            data_2.append(new_entry)

        except Exception as e:
            print(f"Error processing sample {info.get('id')}: {e}")
            continue

    # Save output
    output_path = '/home/as5606/Datasets/Cholec_text_hallucinatory_questions/cholec80_with_new_questions.json'
    print(f"\nSaving {len(data_2)} entries...")
    with open(output_path, 'w') as f:
        json.dump(data_2, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_path}")
    print(f"✓ Processed {len(data_2)} / {len(data)} samples")


if __name__ == "__main__":
    main()
