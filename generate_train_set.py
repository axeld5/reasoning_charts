import json
import os
from datasets import load_dataset
from sentence_level.generate_sample import analyze_text_with_graph, check_answer

def main(sentence_level: bool = True):
    # Create directory for saved graphs if it doesn't exist
    os.makedirs("saved_graphs", exist_ok=True)
    
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"]
    
    # Initialize list to store samples
    samples = []
    
    # Process 100 samples
    for i, example in enumerate(train_data):
        if i < 100:
            # Get the question text
            text = example["question"]
            
            # Generate graph and get analysis
            output_filename = f"saved_graphs/graph_{i}.png"
            question, expected_answer, analysis = analyze_text_with_graph(
                text=text,
                frequency_threshold=3,
                output_filename=output_filename
            )
            
            # Check if answer is valid
            answer_valid = check_answer(expected_answer, analysis)
            
            # Print if answer is invalid
            if not answer_valid:
                print(f"Invalid answer for question: {question}")
            
            # Create sample
            sample = {
                "image": output_filename,
                "question": question,
                "answer": expected_answer,
                "reasoning": analysis,
                "answer_valid": answer_valid
            }
            
            samples.append(sample)
            
            # Print progress
            print(f"Processed sample {i+1}/100")
    
    # Save samples to JSON file
    output_file = "train_set.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Training set saved to {output_file}")

if __name__ == "__main__":
    main() 