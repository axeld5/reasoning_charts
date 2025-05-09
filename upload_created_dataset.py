import json, pathlib
from datasets import Dataset, DatasetDict, Features, Value, Image
from huggingface_hub import login

# 0. authenticate (skipped if you already ran `huggingface-cli login`)
login()

# 1. load your metadata
with open("train_set.json") as f:
    data = json.load(f)

# 2. define the schema
features = Features({
    "image": Image(),           # links to PNG files
    "question": Value("string"),
    "answer": Value("string"),
    "reasoning": Value("string"),
    "answer_valid": Value("bool"),
})

# 3. build a Dataset and wrap in a DatasetDict
ds = Dataset.from_list(data, features=features)
dataset = DatasetDict({"train": ds})

# 4. push!
dataset.push_to_hub(
    "axel-darmouni/anychart-vqa",
    private=False,             # set True if you created a private repo
    commit_message="🚀 Initial upload of images + JSON",
)