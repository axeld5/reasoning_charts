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

def collate_fn_factory(processor, image_token_id, model_name: str, use_reasoning: bool = False):
    """Factory function to create collate_fn for different models."""
    
    def collate_fn(examples):
        texts = []
        images = []
        
        for example in examples:
            image = example["image"]
            if image.mode != 'RGB':
                image = image.convert('RGB')
            question = example["question"]
            question = f"""{question}\n""" + r"Now think and answer the question. Put your answer within $\boxed{}$ tags."
            
            # Use reasoning answers if specified, otherwise use direct answers
            if use_reasoning:
                answer = example["reasoning"] if example["answer_valid"] == True else example["answer"]
            else:
                answer = example["answer"]
            
            if model_name == "qwen2.5_vl_3b":
                # Qwen 2.5 VL format
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
            else:  # SmolVLM format (both 256M and 2.2B)
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
                        {"type": "text", "text": question},
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
        question = f"""{question}\n""" + r"Now think and answer the question. Put your answer within $\boxed{}$ tags."
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
max_samples = test_size  # Use at most 50 test samples for quick evaluation
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

# Prepare training data
train_ds = full_dataset.select(range(train_size))
print(f"Training samples: {len(train_ds)}")

# FINE-TUNING 1: SmolVLM2-256M with non-reasoning answers
print(f"\n{'='*60}")
print("FINE-TUNING SMOLVLM2-256M (NON-REASONING)")
print('='*60)

# Load SmolVLM2-256M for training
model_id_smol = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
model_smol = AutoModelForImageTextToText.from_pretrained(model_id_smol, device_map="auto", torch_dtype=torch.bfloat16).to("cuda")
processor_smol = AutoProcessor.from_pretrained(model_id_smol)
image_token_id_smol = processor_smol.tokenizer.additional_special_tokens_ids[
            processor_smol.tokenizer.additional_special_tokens.index("<image>")]

# Collate function for non-reasoning answers
collate_fn_smol = collate_fn_factory(processor_smol, image_token_id_smol, "smolvlm2_256m", use_reasoning=False)

# Evaluate original SmolVLM2-256M model
print("Evaluating original SmolVLM2-256M model...")
original_results_smol = evaluate_model(model_smol, processor_smol, test_ds, "smolvlm2_256m")
print(f"Original SmolVLM2-256M model accuracy: {original_results_smol['accuracy']:.3f} ({original_results_smol['correct']}/{original_results_smol['total']})")

# Training arguments for SmolVLM
args_smol = TrainingArguments(
    num_train_epochs=10,
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
    output_dir="smolvlm2_256m_nonreasoning",
)

# Train SmolVLM2-256M
trainer_smol = Trainer(
    model=model_smol,
    train_dataset=train_ds,
    data_collator=collate_fn_smol,
    args=args_smol
)

print("Starting SmolVLM2-256M training (non-reasoning)...")
trainer_smol.train()

# Evaluate fine-tuned SmolVLM2-256M model
print("Evaluating fine-tuned SmolVLM2-256M model...")
finetuned_results_smol = evaluate_model(model_smol, processor_smol, test_ds, "smolvlm2_256m")
print(f"Fine-tuned SmolVLM2-256M model accuracy: {finetuned_results_smol['accuracy']:.3f} ({finetuned_results_smol['correct']}/{finetuned_results_smol['total']})")

# Clean up SmolVLM memory before loading Qwen
del model_smol, processor_smol, trainer_smol
torch.cuda.empty_cache()

# FINE-TUNING 2: Qwen2.5 VL 3B with reasoning answers
print(f"\n{'='*60}")
print("FINE-TUNING QWEN2.5-VL-3B (REASONING)")
print('='*60)

# Load Qwen2.5 VL 3B for training
model_id_qwen = "Qwen/Qwen2.5-VL-3B-Instruct"
model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id_qwen, device_map="auto", torch_dtype=torch.bfloat16).to("cuda")
processor_qwen = AutoProcessor.from_pretrained(model_id_qwen)

# Get image token ID for Qwen
image_token_id_qwen = get_image_token_id(processor_qwen, "qwen2.5_vl_3b")

# Collate function for reasoning answers
collate_fn_qwen = collate_fn_factory(processor_qwen, image_token_id_qwen, "qwen2.5_vl_3b", use_reasoning=True)

# Evaluate original Qwen2.5 VL 3B model
print("Evaluating original Qwen2.5 VL 3B model...")
original_results_qwen = evaluate_model(model_qwen, processor_qwen, test_ds, "qwen2.5_vl_3b")
print(f"Original Qwen2.5 VL 3B model accuracy: {original_results_qwen['accuracy']:.3f} ({original_results_qwen['correct']}/{original_results_qwen['total']})")

# Training arguments for Qwen (smaller batch size due to larger model)
args_qwen = TrainingArguments(
    num_train_epochs=5,  # Fewer epochs for larger model
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Larger accumulation for effective batch size
    warmup_steps=2,
    learning_rate=1e-5,  # Lower learning rate for larger model
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=10,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    save_total_limit=1,
    bf16=True,
    output_dir="qwen2_5_vl_3b_reasoning",
)

# Train Qwen2.5 VL 3B
trainer_qwen = Trainer(
    model=model_qwen,
    train_dataset=train_ds,
    data_collator=collate_fn_qwen,
    args=args_qwen
)

print("Starting Qwen2.5 VL 3B training (reasoning)...")
trainer_qwen.train()

# Evaluate fine-tuned Qwen2.5 VL 3B model
print("Evaluating fine-tuned Qwen2.5 VL 3B model...")
finetuned_results_qwen = evaluate_model(model_qwen, processor_qwen, test_ds, "qwen2.5_vl_3b")
print(f"Fine-tuned Qwen2.5 VL 3B model accuracy: {finetuned_results_qwen['accuracy']:.3f} ({finetuned_results_qwen['correct']}/{finetuned_results_qwen['total']})")

# Add fine-tuning results to all_results
all_results["smolvlm2_256m_original"] = original_results_smol
all_results["smolvlm2_256m_finetuned_nonreasoning"] = finetuned_results_smol
all_results["qwen2_5_vl_3b_original"] = original_results_qwen
all_results["qwen2_5_vl_3b_finetuned_reasoning"] = finetuned_results_qwen

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

print(f"\nFINE-TUNING COMPARISON:")
print(f"\nSmolVLM2-256M (Non-Reasoning):")
if "smolvlm2_256m_original" in all_results and "smolvlm2_256m_finetuned_nonreasoning" in all_results:
    orig_smol = all_results["smolvlm2_256m_original"] 
    ft_smol = all_results["smolvlm2_256m_finetuned_nonreasoning"]
    improvement_smol = ft_smol['accuracy'] - orig_smol['accuracy']
    print(f"  Original: {orig_smol['accuracy']:.3f} ({orig_smol['correct']}/{orig_smol['total']})")
    print(f"  Fine-tuned: {ft_smol['accuracy']:.3f} ({ft_smol['correct']}/{ft_smol['total']})")
    print(f"  Improvement: {improvement_smol:.3f} ({improvement_smol*100:.1f}%)")

print(f"\nQwen2.5 VL 3B (Reasoning):")
if "qwen2_5_vl_3b_original" in all_results and "qwen2_5_vl_3b_finetuned_reasoning" in all_results:
    orig_qwen = all_results["qwen2_5_vl_3b_original"] 
    ft_qwen = all_results["qwen2_5_vl_3b_finetuned_reasoning"]
    improvement_qwen = ft_qwen['accuracy'] - orig_qwen['accuracy']
    print(f"  Original: {orig_qwen['accuracy']:.3f} ({orig_qwen['correct']}/{orig_qwen['total']})")
    print(f"  Fine-tuned: {ft_qwen['accuracy']:.3f} ({ft_qwen['correct']}/{ft_qwen['total']})")
    print(f"  Improvement: {improvement_qwen:.3f} ({improvement_qwen*100:.1f}%)")

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
        "finetuning_experiments": {
            "SmolVLM2-256M": "Fine-tuned with non-reasoning answers (direct answers)",
            "Qwen-2.5-VL-3B": "Fine-tuned with reasoning answers (step-by-step explanations)"
        },
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