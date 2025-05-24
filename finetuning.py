import torch
import json
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, BitsAndBytesConfig
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import numpy as np

# Import checking functions from generate_train_set modules
from sentence_level.generate_barplot_sample import check_barplot_answer
from sentence_level.generate_piechart_sample import check_piechart_answer
from corpus_level.generate_wordcloud_sample import check_wordcloud_answer

load_dotenv()
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" 

device = "cuda:0"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)
image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]

def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image = example["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        question = example["question"]
        if example["answer_valid"] == "true":
            answer = example["reasoning"]
        else:
            answer = example["answer"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

def generate_response(model, processor, image: Image.Image, question: str) -> str:
    """Generate a response from the model for a given image and question."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
        )
    
    # Decode only the generated part
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )[0]
    
    return generated_text.strip()

def check_answer_validity(expected_answer: str, generated_answer: str, chart_type: str) -> bool:
    """Check if the generated answer is valid using the appropriate checking function."""
    if chart_type == "barplot":
        return check_barplot_answer(expected_answer, generated_answer)
    elif chart_type == "piechart":
        # For piechart, we need the answer_type, but since we don't have it, 
        # we'll assume it's a count-based question
        return check_piechart_answer(expected_answer, generated_answer, "count")
    elif chart_type == "wordcloud":
        return check_wordcloud_answer(expected_answer, generated_answer)
    else:
        # Fallback: simple string comparison
        return expected_answer.lower().strip() in generated_answer.lower().strip()

def evaluate_model(model, processor, test_dataset) -> Dict[str, float]:
    """Evaluate a model on the test dataset."""
    correct = 0
    total = 0
    
    for example in test_dataset:
        image = example["image"]
        question = example["question"]
        expected_answer = example["answer"]
        chart_type = example["chart_type"]
        
        generated_answer = generate_response(model, processor, image, question)
        is_correct = check_answer_validity(expected_answer, generated_answer, chart_type)
        
        if is_correct:
            correct += 1
        total += 1
        
        print(f"Question: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Generated: {generated_answer}")
        print(f"Correct: {is_correct}")
        print("-" * 50)
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

# Load and split dataset
full_dataset = load_dataset("axel-darmouni/anychart-vqa")["train"]

# Split into train (80%) and test (20%)
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

# Use a smaller subset for faster training/testing during development
max_samples = min(100, dataset_size)  # Use at most 100 samples
train_size = int(0.8 * max_samples)
test_size = max_samples - train_size

train_ds = full_dataset.select(range(train_size))
test_ds = full_dataset.select(range(train_size, train_size + test_size))

print(f"Training samples: {len(train_ds)}")
print(f"Test samples: {len(test_ds)}")

# Evaluate original model
print("Evaluating original model...")
original_results = evaluate_model(model, processor, test_ds)
print(f"Original model accuracy: {original_results['accuracy']:.3f} ({original_results['correct']}/{original_results['total']})")

# Training arguments
args = TrainingArguments(
    num_train_epochs=3,  # Reduced for faster training
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=10,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    save_total_limit=1,
    output_dir="smolvlm2_ocr_thinking",
    dataloader_pin_memory=False,
    evaluation_strategy="no",  # We'll do manual evaluation
)

# Train the model
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    data_collator=collate_fn,
    args=args
)

print("Starting training...")
trainer.train()

# Evaluate fine-tuned model
print("Evaluating fine-tuned model...")
finetuned_results = evaluate_model(model, processor, test_ds)
print(f"Fine-tuned model accuracy: {finetuned_results['accuracy']:.3f} ({finetuned_results['correct']}/{finetuned_results['total']})")

# Print comparison
print("\n" + "="*60)
print("EVALUATION RESULTS COMPARISON")
print("="*60)
print(f"Original model accuracy: {original_results['accuracy']:.3f} ({original_results['correct']}/{original_results['total']})")
print(f"Fine-tuned model accuracy: {finetuned_results['accuracy']:.3f} ({finetuned_results['correct']}/{finetuned_results['total']})")
improvement = finetuned_results['accuracy'] - original_results['accuracy']
print(f"Improvement: {improvement:.3f} ({improvement*100:.1f}%)")

# Save results to file
results = {
    "original_model": original_results,
    "finetuned_model": finetuned_results,
    "improvement": improvement,
    "train_samples": len(train_ds),
    "test_samples": len(test_ds)
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to evaluation_results.json")