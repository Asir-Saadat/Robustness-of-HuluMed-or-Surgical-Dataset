#!/usr/bin/env python3
"""
Generate semantically similar question variations for Cholec80 surgical VQA dataset.
Uses OpenAI API to create rephrasings of existing questions.
"""

import json
import os
import re
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client (use environment variable for API key)
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=API_KEY)

# System prompt
REPHRASE_PROMPT = """\
You rephrase questions for a surgical visual-question-answering benchmark built
on laparoscopic cholecystectomy (Cholec80) video frames. Given ONE question,
produce several SEMANTICALLY EQUIVALENT variations: same meaning, same answer,
different surface form.

The purpose is to test whether a model answers CONSISTENTLY regardless of how the
question is worded. Therefore the answer MUST NOT change across your variations.

PRESERVE EXACTLY - never alter, drop, or replace with synonyms:
- the tool / instrument name (e.g. "specimen bag", "hook", "grasper")
- the phase name (e.g. "calot triangle dissection", "clipping and cutting")
- the answer type: a yes/no question stays yes/no; a counting question stays a
  counting question. NEVER turn "is X used in Y" into "what is X doing in Y" -
  that changes the answer.

VARY FREELY:
- wording, grammar, active/passive voice, sentence structure, clinical phrasing,
  question framing ("Is ...", "Does ...", "During ..., is ...").

RULES:
- Every variation must read naturally, as a surgeon might ask.
- No two variations may be identical or differ only in punctuation.
- Do NOT add information or assumptions not in the original.

EXAMPLES
Original: "is specimen bag used in calot triangle dissection"
- "During calot triangle dissection, is the specimen bag used?"
- "Is the specimen bag employed in the calot triangle dissection phase?"
- "Does calot triangle dissection involve the use of the specimen bag?"

Original: "how many tools are operating?"
- "What is the number of tools currently in operation?"
- "How many tools are in use at this moment?"
- "Count the tools that are currently operating."

OUTPUT - return ONLY valid JSON, no preamble, no markdown fences:
{"variations": ["...", "...", ...]}
"""


def _parse_json(raw: str) -> dict:
    """Strip accidental code fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def extract_tool_phase(question: str) -> tuple:
    """Extract tool and phase from yes/no question format."""
    # Pattern: "is <tool> used in <phase>"
    match = re.search(r'is\s+(.+?)\s+used\s+in\s+(.+?)(?:\?)?$', question, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def rephrase_question(question: str, n: int = 5,
                      tool: str | None = None, phase: str | None = None) -> list[str]:
    """Generate `n` semantically-equivalent rephrasings of `question`.

    If `tool` and `phase` are given, they are (a) emphasized in the prompt and
    (b) used to drop any variation that mangled them.
    """
    anchor = ""
    if tool and phase:
        anchor = f'Keep the tool name "{tool}" and the phase name "{phase}" verbatim.\n'

    user_msg = (
        anchor
        + f'Original question: "{question}"\n'
        + f"Generate {n} semantically equivalent variations."
    )

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=user_msg,
        instructions=REPHRASE_PROMPT,
    )

    try:
        output_text = response.output[0].content[0].text
        variations = _parse_json(output_text)["variations"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Parse failed for: {question!r}: {e}")
        return []

    # Guard against drift: if anchors were given, keep only variations that
    # still contain both the tool and the phase (case-insensitive).
    if tool and phase:
        t, p = tool.lower(), phase.lower()
        variations = [v for v in variations if t in v.lower() and p in v.lower()]

    return variations


def main():
    """Main processing function."""
    print("Loading test dataset...")
    with open('/home/as5606/Datasets/cholec_formatted_data/cholec80_llava_test.json', 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    data_2 = []

    print("Processing samples...")
    for info in tqdm(data, desc="Processing"):
        question = info['conversations'][0]['value'].replace('\n<image>', '').strip()
        answer = info['conversations'][1]['value']

        new_entry = {**info}  # Copy original entry

        try:
            # Extract tool and phase for anchoring (if yes/no question)
            tool, phase = extract_tool_phase(question)

            # Generate variations
            variations = rephrase_question(question, n=5, tool=tool, phase=phase)

            if variations:
                # Add variations to entry
                new_entry['semantic_variations'] = variations
                data_2.append(new_entry)

        except Exception as e:
            print(f"Error processing sample {info.get('id')}: {e}")
            continue

    # Save output
    output_path = '/home/as5606/Datasets/Cholec_text_semantic_similar_questions/cholec80_with_semantic_variations.json'
    print(f"\nSaving {len(data_2)} entries...")
    with open(output_path, 'w') as f:
        json.dump(data_2, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_path}")
    print(f"✓ Processed {len(data_2)} / {len(data)} samples")


if __name__ == "__main__":
    main()
