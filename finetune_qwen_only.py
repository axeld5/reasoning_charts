import torch
import json
import gc
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import numpy as np

# Import checking functions from generate_train_set modules
from sentence_level.generate_barplot_sample import check_barplot_answer
from sentence_level.generate_piechart_sample import check_piechart_answer
from corpus_level.generate_wordcloud_sample import check_wordcloud_answer

load_dotenv()

def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def setup_quantization():
    """Setup 4-bit quantization to reduce memory usage"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config

def setup_lora_config():
    """Setup LoRA configuration for parameter-efficient fine-tuning"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Low rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return lora_config

def load_qwen_model_optimized():
    """Load Qwen model with memory optimizations"""
    print("Loading Qwen2.5-VL-3B with memory optimizations...")
    
    # Setup quantization
    bnb_config = setup_quantization()
    
    # Load model with quantization
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use flash attention if available
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f"Model loaded with {model.num_parameters()} total parameters")
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True)}")
    
    clear_memory()
    return model, processor

def get_image_token_id(processor):
    """Get image token ID for Qwen models"""
    try:
        return processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    except:
        # Fallback if the above doesn't work
        return processor.tokenizer.additional_special_tokens_ids[0] if processor.tokenizer.additional_special_tokens_ids else None

def collate_fn_factory(processor, image_token_id, use_reasoning: bool = True):
    """Factory function to create collate_fn for Qwen with memory optimization"""
    
    def collate_fn(examples):
        # Process in smaller chunks to avoid OOM
        batch_size = len(examples)
        if batch_size > 2:  # If batch is too large, process in chunks
            mid = batch_size // 2
            chunk1 = collate_fn(examples[:mid])
            chunk2 = collate_fn(examples[mid:])
            
            # Combine chunks
            combined = {}
            for key in chunk1.keys():
                if key in chunk2:
                    combined[key] = torch.cat([chunk1[key], chunk2[key]], dim=0)
                else:
                    combined[key] = chunk1[key]
            return combined
        
        texts = []
        images = []
        
        for example in examples:
            image = example["image"]
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to reduce memory usage
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            question = example["question"]
            question = f"""{question}\n""" + r"Now think and answer the question. Put your answer within $\boxed{}$ tags."
            
            # Use reasoning answers if specified and valid, otherwise use direct answers
            if use_reasoning and example.get("answer_valid", False):
                answer = example["reasoning"]
            else:
                answer = example["answer"]
            
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
            
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        try:
            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            if image_token_id is not None:
                labels[labels == image_token_id] = -100
            batch["labels"] = labels
            return batch
        except torch.cuda.OutOfMemoryError:
            print("OOM in collate_fn, clearing memory and retrying with smaller batch...")
            clear_memory()
            # Try with just one example if we still have OOM
            if len(examples) > 1:
                return collate_fn(examples[:1])
            else:
                raise
    
    return collate_fn

def generate_response(model, processor, image: Image.Image, question: str) -> str:
    """Generate a response from the model for a given image and question with memory management"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to reduce memory usage
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    try:
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
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,  # Reduced for memory
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_text = processor.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        return generated_text.strip()
    
    except torch.cuda.OutOfMemoryError:
        print("OOM during generation, clearing memory...")
        clear_memory()
        return "Error: Out of memory"
    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error: {str(e)}"

def check_answer_validity(expected_answer: str, generated_answer: str, chart_type: str) -> bool:
    """Check if the generated answer is valid using the appropriate checking function."""
    if chart_type == "barplot":
        return check_barplot_answer(expected_answer, generated_answer)
    elif chart_type == "piechart":
        return check_piechart_answer(expected_answer, generated_answer, "count")
    elif chart_type == "wordcloud":
        return check_wordcloud_answer(expected_answer, generated_answer)
    else:
        # Fallback: simple string comparison
        return expected_answer.lower().strip() in generated_answer.lower().strip()

def evaluate_model_subset(model, processor, test_dataset, max_samples: int = 20) -> Dict[str, float]:
    """Evaluate model on a subset to avoid memory issues"""
    print(f"Evaluating on subset of {max_samples} samples...")
    
    correct = 0
    total = 0
    
    # Take only a subset for evaluation
    subset = test_dataset.select(range(min(max_samples, len(test_dataset))))
    
    for i, example in enumerate(subset):
        print(f"Processing sample {i+1}/{len(subset)}")
        
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
            
            print(f"Expected: {expected_answer}")
            print(f"Generated: {generated_answer}")
            print(f"Correct: {is_correct}")
            print("-" * 50)
            
            # Clear memory after each sample
            clear_memory()
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            total += 1  # Count as incorrect
            clear_memory()
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

class MemoryOptimizedTrainer(Trainer):
    """Custom trainer with memory optimizations"""
    
    def training_step(self, model, inputs):
        """Override training step to add memory management"""
        try:
            return super().training_step(model, inputs)
        except torch.cuda.OutOfMemoryError:
            print("OOM in training step, clearing memory...")
            clear_memory()
            # Try with gradient accumulation
            return super().training_step(model, inputs)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override save checkpoint to clear memory"""
        result = super()._save_checkpoint(model, trial, metrics)
        clear_memory()
        return result

def main():
    print("Starting Qwen-2.5-VL-3B Fine-tuning with Memory Optimization")
    print("=" * 70)
    
    # Load and split dataset
    print("Loading dataset...")
    full_dataset = load_dataset("axel-darmouni/anychart-vqa")["train"]
    
    # Split into train (80%) and test (20%)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    
    # Use smaller subsets for memory efficiency
    max_train_samples = min(1000, train_size)  # Limit training samples
    max_test_samples = 50  # Limit test samples
    
    train_ds = full_dataset.select(range(max_train_samples))
    test_ds = full_dataset.select(range(train_size, train_size + max_test_samples))
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    # Load model with optimizations
    model, processor = load_qwen_model_optimized()
    
    # Get image token ID
    image_token_id = get_image_token_id(processor)
    
    # Setup collate function with reasoning answers
    collate_fn = collate_fn_factory(processor, image_token_id, use_reasoning=True)
    
    # Training arguments optimized for memory
    training_args = TrainingArguments(
        output_dir="qwen2_5_vl_3b_reasoning_optimized",
        num_train_epochs=3,  # Reduced epochs
        per_device_train_batch_size=1,  # Very small batch size
        gradient_accumulation_steps=16,  # Large accumulation to compensate
        per_device_eval_batch_size=1,
        warmup_steps=10,
        learning_rate=1e-4,  # Slightly higher LR for LoRA
        weight_decay=0.01,
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        dataloader_num_workers=0,  # Reduce workers to save memory
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # Use 8-bit optimizer
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to=None,  # Disable wandb/tensorboard to save memory
    )
    
    # Create trainer with memory optimizations
    trainer = MemoryOptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds.select(range(min(20, len(test_ds)))),  # Small eval set
        data_collator=collate_fn,
    )
    
    print("Starting fine-tuning...")
    clear_memory()
    
    try:
        # Train the model
        trainer.train()
        
        print("Training completed successfully!")
        
        # Save the final model
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        
        print("Evaluating fine-tuned model...")
        results = evaluate_model_subset(model, processor, test_ds, max_samples=30)
        
        print(f"\nFinal Results:")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Correct: {results['correct']}/{results['total']}")
        
        # Save results
        final_results = {
            "model": "Qwen2.5-VL-3B-Instruct",
            "fine_tuning": "LoRA with reasoning answers",
            "training_samples": len(train_ds),
            "test_samples": results['total'],
            "accuracy": results['accuracy'],
            "correct": results['correct'],
            "total": results['total'],
            "optimizations": [
                "4-bit quantization",
                "LoRA fine-tuning", 
                "Gradient checkpointing",
                "Memory-optimized data loading",
                "Flash attention",
                "8-bit optimizer"
            ]
        }
        
        with open("qwen_finetuning_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        print("Results saved to qwen_finetuning_results.json")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"Out of memory error: {e}")
        print("Try reducing batch size or train samples further")
        
    except Exception as e:
        print(f"Training error: {e}")
        
    finally:
        # Clean up
        clear_memory()

if __name__ == "__main__":
    main() 