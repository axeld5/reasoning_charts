import json
import os
import random
from datasets import load_dataset
from sentence_level.generate_barplot_sample import analyze_text_with_barplot, check_barplot_answer
from sentence_level.generate_piechart_sample import analyze_text_with_piechart, check_piechart_answer
from corpus_level.generate_wordcloud_sample import analyze_text_with_wordcloud, check_wordcloud_answer

def main(sentence_level: bool = False, corpus_level: bool = True, dataset_name: str = "gsm8k", dataset_split: str = "main"):
    # Create directory for saved graphs if it doesn't exist
    os.makedirs("saved_graphs", exist_ok=True)
    
    # Load specified dataset
    dataset = load_dataset(dataset_name, dataset_split)
    train_data = dataset["train"]
    
    # Initialize list to store samples
    samples = []
    
    # Process 100 samples
    if sentence_level:
        for i, example in enumerate(train_data):
        
            if i < 100:
                # Get the question text
                text = example["question"]
                
                # Generate graph and get analysis
                output_filename = f"saved_graphs/graph_{i}.png"
                question, expected_answer, analysis = analyze_text_with_barplot(
                    text=text,
                    frequency_threshold=3,
                    output_filename=output_filename
                )
                
                # Check if answer is valid
                answer_valid = check_barplot_answer(expected_answer, analysis)
                
                # Print if answer is invalid
                if not answer_valid:
                    print(f"Invalid answer for question: {question}")
                
                # Create sample
                sample = {
                    "image": output_filename,
                    "question": question,
                    "answer": expected_answer,
                    "reasoning": analysis,
                    "answer_valid": answer_valid,
                    "chart_type": "barplot"
                }
                
                samples.append(sample)
                
                # Print progress
                print(f"Processed sample {i+1}/100")
            elif i < 200:
                text = example["question"]
                
                # Generate graph and get analysis
                output_filename = f"saved_graphs/graph_{i}.png"
                question, expected_answer, answer_type, analysis = analyze_text_with_piechart(
                    text=text,
                    frequency_threshold=3,
                    output_filename=output_filename
                )
                
                # Check if answer is valid
                answer_valid = check_piechart_answer(expected_answer, analysis, answer_type)
                
                # Print if answer is invalid
                if not answer_valid:
                    print(f"Invalid answer for question: {question}")
                
                # Create sample
                sample = {
                    "image": output_filename,
                    "question": question,
                    "answer": expected_answer,
                    "reasoning": analysis,
                    "answer_valid": answer_valid,
                    "chart_type": "piechart"
                }
                
                samples.append(sample)
                
                # Print progress
                print(f"Processed sample {i+1}/200")
    elif corpus_level:
        for i in range(100):            
            # Get random samples from the training set
            random_indices = random.sample(range(len(train_data)), 50)
            sample_texts = [train_data[idx]["question"] for idx in random_indices]
            
            # Generate wordcloud and get analysis
            output_filename = f"saved_graphs/wordcloud_{i}.png"
            question, expected_answer, analysis = analyze_text_with_wordcloud(
                texts=sample_texts,
                frequency_threshold=10,
                output_filename=output_filename
            )
            
            # Check if answer is valid
            answer_valid = check_wordcloud_answer(expected_answer, analysis)
            
            # Print if answer is invalid
            if not answer_valid:
                print(f"Invalid answer for question: {question}")
            
            # Create sample
            sample = {
                "image": output_filename,
                "question": question,
                "answer": expected_answer,
                "reasoning": analysis,
                "answer_valid": answer_valid,
                "chart_type": "wordcloud"
            }
            
            samples.append(sample)
            
            # Print progress
            print(f"Processed wordcloud sample {i+1}/100")
    
    # Save samples to JSON file
    output_file = "train_set_v3.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Training set saved to {output_file}")

if __name__ == "__main__":
    main(dataset_name="gsm8k", dataset_split="main") 