#!/usr/bin/env python3
"""
Hulu-Med analysis script for original Cholec80 test dataset (answerable questions).
Generates Hulu-Med answers for original answerable surgical questions.
Supports 4B, 7B, and 14B models.
"""

import torch
import sys
from transformers import AutoModelForCausalLM, AutoProcessor
import json
from tqdm import tqdm

# Get model size from command line argument
if len(sys.argv) < 2:
    print("Usage: python3 hulu_med_hallucinatory_answerable.py <model_size>")
    print("Example: python3 hulu_med_hallucinatory_answerable.py 4B")
    sys.exit(1)

model_size = sys.argv[1]
print(f"Running analysis for Hulu-Med-{model_size}")

# Map model sizes to paths
MODEL_PATHS = {
    "4B": "ZJU-AI4H/Hulu-Med-4B",
    "7B": "ZJU-AI4H/Hulu-Med-7B",
    "14B": "ZJU-AI4H/Hulu-Med-14B",
}

if model_size not in MODEL_PATHS:
    print(f"Error: Model size {model_size} not supported. Choose from: 4B, 7B, 14B")
    sys.exit(1)

MODEL_PATH = MODEL_PATHS[model_size]

# Load model and processor
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device_map = "auto" if torch.cuda.is_available() else None

print(f"Loading Hulu-Med-{model_size} model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map=device_map,
    attn_implementation="sdpa",
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
print("Loading original test dataset...")
with open('/home/as5606/Datasets/cholec_formatted_data/cholec80_llava_test.json', 'r') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples")

# Store results
results = []

print(f"Generating Hulu-Med-{model_size} answers for original questions...")
for item in tqdm(test_data, desc="Processing"):
    # Extract data
    item_id = item['id']
    frame_path = item['image']
    original_question = item['conversations'][0]['value'].replace('\n<image>', '').strip()
    original_answer = item['conversations'][1]['value']

    try:
        # Add instruction to question
        question_with_instruction = original_question + " If you don't know the answer, say 'I don't know'."

        # Create conversation for original question
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
                        "text": question_with_instruction,
                    },
                ],
            }
        ]

        # Generate Hulu-Med answer
        hulu_answer = hulumed_generate(conversation, modal="image", max_new_tokens=256)

        # Store result
        result = {
            "id": item_id,
            "image": frame_path,
            "original_question": original_question,
            "original_answer": original_answer,
            "hulu_med_answer": hulu_answer
        }
        results.append(result)

    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        continue

# Save to JSON
output_path = f'/home/as5606/Datasets/Cholec_text_hallucinatory_questions/hulu_med_{model_size}_answerable_answers.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(results)} results to {output_path}")
