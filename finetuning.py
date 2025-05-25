import torch
import json
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import numpy as np

# Import checking functions from generate_train_set modules
from sentence_level.generate_barplot_sample import check_barplot_answer
from sentence_level.generate_piechart_sample import check_piechart_answer
from corpus_level.generate_wordcloud_sample import check_wordcloud_answer

load_dotenv()

# Model configurations
MODELS_CONFIG = {
    "smolvlm2_256m": {
        "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "model_class": AutoModelForImageTextToText,
        "processor_class": AutoProcessor,
        "supports_vision": True
    },
    "smolvlm2_2.2b": {
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "model_class": AutoModelForImageTextToText,
        "processor_class": AutoProcessor,
        "supports_vision": True
    },
    "qwen2.5_vl_3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", 
        "model_class": Qwen2_5_VLForConditionalGeneration,
        "processor_class": AutoProcessor,
        "supports_vision": True
    }
}

device = "cuda:0"

def load_model_and_processor(model_name: str):
    """Load model and processor based on model configuration."""
    config = MODELS_CONFIG[model_name]
    
    if not config["supports_vision"]:
        raise ValueError(f"Model {model_name} ({config['model_id']}) does not support vision tasks!")
    
    print(f"Loading {model_name}: {config['model_id']}")
    
    model = config["model_class"].from_pretrained(
        config["model_id"], 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    processor = config["processor_class"].from_pretrained(config["model_id"])
    
    return model, processor

def get_image_token_id(processor, model_name: str):
    """Get image token ID for different model types."""
    if model_name == "qwen2.5_vl_3b":
        # Qwen models may use different special tokens
        try:
            return processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        except:
            # Fallback if the above doesn't work
            return processor.tokenizer.additional_special_tokens_ids[0] if processor.tokenizer.additional_special_tokens_ids else None
    else:  # SmolVLM (both 256M and 2.2B versions)
        return processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]

def collate_fn_factory(processor, image_token_id, model_name: str):
    """Factory function to create collate_fn for different models."""
    
    def collate_fn(examples):
        texts = []
        images = []
        
        for example in examples:
            image = example["image"]
            if image.mode != 'RGB':
                image = image.convert('RGB')
            question = example["question"]
            if example["answer_valid"] == True:
                answer = example["reasoning"]
            else:
                answer = example["answer"]
            
            if model_name == "qwen2.5_vl_3b":
                # Qwen 2.5 VL format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"""{question}\n""" + r"Now think and answer the question. Put your answer within $\boxed{}$ tags."},
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
            else:  # SmolVLM format (both 256M and 2.2B)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"""{question}\n""" + r"Now think and answer the question. Put your answer within $\boxed{}$ tags."},
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
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch
    
    return collate_fn

def generate_response_factory(model_name: str):
    """Factory function to create response generation function for different models."""
    
    def generate_response(model, processor, image: Image.Image, question: str) -> str:
        """Generate a response from the model for a given image and question."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if model_name == "qwen2.5_vl_3b":
            # Qwen 2.5 VL format
            from qwen_vl_utils import process_vision_info
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"""{question}\n""" + r"Now think and answer the question. Put your answer within $\boxed{}$ tags."},
                        {"type": "image", "image": image},
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            
        else:  # SmolVLM format (both 256M and 2.2B)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"""{question}\n""" + r"Now think and answer the question. Put your answer within $\boxed{}$ tags."},
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
    
    return generate_response

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

def evaluate_model(model, processor, test_dataset, model_name: str) -> Dict[str, float]:
    """Evaluate a model on the test dataset."""
    generate_response = generate_response_factory(model_name)
    
    correct = 0
    total = 0
    
    print(f"\nEvaluating {model_name}...")
    
    for example in test_dataset:
        image = example["image"]
        question = example["question"]
        expected_answer = example["answer"]
        chart_type = example["chart_type"]
        
        try:
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
            
        except Exception as e:
            print(f"Error processing example: {e}")
            total += 1  # Count as incorrect
    
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

# Use a smaller subset for faster evaluation during development
max_samples = min(50, test_size)  # Use at most 50 test samples for quick evaluation
test_ds = full_dataset.select(range(train_size, train_size + max_samples))

print(f"Test samples: {len(test_ds)}")

# Initialize results dictionary
all_results = {}

# Evaluate all supported models
for model_name in MODELS_CONFIG.keys():
    config = MODELS_CONFIG[model_name]
    
    if not config["supports_vision"]:
        print(f"\nSkipping {model_name} ({config['model_id']}) - No vision support")
        continue
    
    try:
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print('='*60)
        
        # Load model and processor
        model, processor = load_model_and_processor(model_name)
        
        # Evaluate the model
        results = evaluate_model(model, processor, test_ds, model_name)
        all_results[model_name] = results
        
        print(f"{model_name} accuracy: {results['accuracy']:.3f} ({results['correct']}/{results['total']})")
        
        # Clean up memory
        del model, processor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        all_results[model_name] = {"error": str(e)}

# Fine-tune SmolVLM2-256M (original training code)
print(f"\n{'='*60}")
print("FINE-TUNING SMOLVLM2-256M")
print('='*60)

# Reload SmolVLM2-256M for training
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)
image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]

# Prepare training data
train_ds = full_dataset.select(range(train_size))
collate_fn = collate_fn_factory(processor, image_token_id, "smolvlm2_256m")

print(f"Training samples: {len(train_ds)}")

# Evaluate original SmolVLM2-256M model
print("Evaluating original SmolVLM2-256M model...")
original_results = evaluate_model(model, processor, test_ds, "smolvlm2_256m")
print(f"Original SmolVLM2-256M model accuracy: {original_results['accuracy']:.3f} ({original_results['correct']}/{original_results['total']})")

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
    bf16=True,
    output_dir="smolvlm2_256m_chart_thinking",
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
print("Evaluating fine-tuned SmolVLM2-256M model...")
finetuned_results = evaluate_model(model, processor, test_ds, "smolvlm2_256m")
print(f"Fine-tuned SmolVLM2-256M model accuracy: {finetuned_results['accuracy']:.3f} ({finetuned_results['correct']}/{finetuned_results['total']})")

# Add fine-tuning results to all_results
all_results["smolvlm2_256m_original"] = original_results
all_results["smolvlm2_256m_finetuned"] = finetuned_results

# Print comprehensive comparison
print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION RESULTS")
print("="*80)

print("\nPRE-TRAINED MODEL COMPARISON:")
for model_name, results in all_results.items():
    if "error" in results:
        print(f"{model_name}: ERROR - {results['error']}")
    elif model_name not in ["smolvlm2_256m_original", "smolvlm2_256m_finetuned"]:
        print(f"{model_name}: {results['accuracy']:.3f} ({results['correct']}/{results['total']})")

print(f"\nFINE-TUNING COMPARISON (SmolVLM2-256M):")
if "smolvlm2_256m_original" in all_results and "smolvlm2_256m_finetuned" in all_results:
    orig = all_results["smolvlm2_256m_original"] 
    ft = all_results["smolvlm2_256m_finetuned"]
    improvement = ft['accuracy'] - orig['accuracy']
    print(f"Original SmolVLM2-256M: {orig['accuracy']:.3f} ({orig['correct']}/{orig['total']})")
    print(f"Fine-tuned SmolVLM2-256M: {ft['accuracy']:.3f} ({ft['correct']}/{ft['total']})")
    print(f"Improvement: {improvement:.3f} ({improvement*100:.1f}%)")

print(f"\nMODEL CAPABILITIES:")
print("• SmolVLM2-256M: Lightweight multimodal support ✓")
print("• SmolVLM2-2.2B: Enhanced multimodal support ✓")
print("• Qwen-2.5-VL-3B: Full multimodal support ✓")

# Save comprehensive results to file
final_results = {
    "evaluation_models": all_results,
    "test_samples": len(test_ds),
    "notes": {
        "supported_models": ["SmolVLM2-256M", "SmolVLM2-2.2B", "Qwen-2.5-VL-3B"],
        "dataset": "axel-darmouni/anychart-vqa",
        "model_details": {
            "SmolVLM2-256M": "Lightweight multimodal model for efficient inference",
            "SmolVLM2-2.2B": "Enhanced multimodal model with better performance",
            "Qwen-2.5-VL-3B": "Larger multimodal model with advanced capabilities"
        }
    }
}

with open("comprehensive_evaluation_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\nComprehensive results saved to comprehensive_evaluation_results.json")