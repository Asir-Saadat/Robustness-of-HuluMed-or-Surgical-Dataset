#!/usr/bin/env python3
"""
Hulu-Med-7B hallucinatory questions analysis script for Cholec80 dataset.
Generates Hulu-Med answers for unanswerable/hallucinatory surgical questions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import json
from tqdm import tqdm

# Load model and processor
MODEL_PATH = "ZJU-AI4H/Hulu-Med-7B"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device_map = "auto" if torch.cuda.is_available() else None

print("Loading Hulu-Med-7B model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map=device_map,
    attn_implementation="sdpa",   # safer than flash_attention_2 unless flash-attn is installed
)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

tokenizer = processor.tokenizer
model.eval()

print("Model loaded successfully!")


def hulumed_generate(conversation, modal="text", max_new_tokens=1024, temperature=0.0):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if "modals" not in inputs:
        inputs["modals"] = [modal]

    model_device = next(model.parameters()).device
    use_cuda = model_device.type == "cuda"

    fixed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                dtype = torch.bfloat16 if use_cuda else torch.float32
                fixed_inputs[k] = v.to(device=model_device, dtype=dtype)
            else:
                fixed_inputs[k] = v.to(model_device)
        else:
            fixed_inputs[k] = v

    generation_kwargs = {
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(
            **fixed_inputs,
            **generation_kwargs,
        )

    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        use_think=False,
    )[0].strip()


# Load test dataset
print("Loading test dataset...")
with open('/home/as5606/Datasets/Cholec_text_hallucinatory_questions/cholec80_with_new_questions.json', 'r') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples")

# Store results
results = []

print("Generating Hulu-Med answers for new questions...")
for item in tqdm(test_data, desc="Processing"):
    # Extract data
    item_id = item['id']
    frame_path = item['image']
    original_question = item['conversations'][0]['value'].replace('\n<image>', '').strip()
    original_answer = item['conversations'][1]['value']

    # Extract the new questions
    new_questions = item.get('new_questions', [])

    try:
        # Generate Hulu-Med answers for each new question
        new_questions_with_hulu = []
        for new_q in new_questions:
            new_question = new_q['question']+" If the question cannot be answered based on the image, say 'I don't know'."

            # Create conversation for new question
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": {
                                "image_path": frame_path,
                            },
                        },
                        {
                            "type": "text",
                            "text": new_question,
                        },
                    ],
                }
            ]

            # Generate Hulu-Med answer for new question
            hulu_answer = hulumed_generate(conversation, modal="image", max_new_tokens=256)
            new_questions_with_hulu.append({
                "type": new_q.get('type', ''),
                "question": new_question,
                "hulu_med_answer": hulu_answer
            })

        # Store result with original data + new questions with Hulu-Med answers
        result = {
            "id": item_id,
            "image": frame_path,
            "original_question": original_question,
            "original_answer": original_answer,
            "new_questions_with_hulu_answers": new_questions_with_hulu
        }
        results.append(result)

    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = '/home/as5606/Datasets/Cholec_text_hallucinatory_questions/hulu_med_7B_hallucinatory_answers.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")
print(f"Sample result: {results[0] if results else 'No results'}")
