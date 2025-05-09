import json, pathlib
from datasets import Dataset, DatasetDict, Features, Value, Image
from huggingface_hub import login
import glob
import argparse

def load_data_from_json(json_file):
    """Load and process data from a single JSON file."""
    with open(json_file) as f:
        file_data = json.load(f)
        # If the file is in train_data directory, extract chart_type from filename
        if "train_data" in json_file:
            chart_type = json_file.split("\\")[1].split("_")[0]
            for item in file_data:
                item["chart_type"] = chart_type
        return file_data

def main(input_path: str, repo_name: str = "axel-darmouni/anychart-vqa"):
    # 0. authenticate (skipped if you already ran `huggingface-cli login`)
    login()

    # 1. load your metadata
    data = []
    if input_path.endswith('.json'):
        # Single JSON file
        print(f"Loading data from {input_path}")
        data.extend(load_data_from_json(input_path))
    else:
        # Directory containing multiple JSON files
        print(f"Loading data from directory: {input_path}")
        for json_file in glob.glob(f"{input_path}/*.json"):
            print(f"Processing {json_file}")
            data.extend(load_data_from_json(json_file))

    # 2. define the schema
    features = Features({
        "image": Image(),           # links to PNG files
        "question": Value("string"),
        "answer": Value("string"),
        "reasoning": Value("string"),
        "answer_valid": Value("bool"),
        "chart_type": Value("string"),
    })

    # 3. build a Dataset and wrap in a DatasetDict
    ds = Dataset.from_list(data, features=features)
    dataset = DatasetDict({"train": ds})

    # 4. push!
    dataset.push_to_hub(
        repo_name,
        private=False,
        commit_message="🚀 Initial upload of images + JSON with chart types",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload dataset to Hugging Face Hub')
    parser.add_argument('--input_path', type=str, required=True, default="train_data",
                      help='Path to either a single JSON file or directory containing JSON files')
    parser.add_argument('--repo_name', type=str, default="axel-darmouni/anychart-vqa",
                      help='Hugging Face repository name')
    
    args = parser.parse_args()
    main(args.input_path, args.repo_name)