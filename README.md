# Reasoning Charts

A Python project that generates and analyzes various types of charts from text, with AI-powered question answering capabilities.

## Overview

This project analyzes text input to create different types of visualizations (bar charts, pie charts, and word clouds) and uses Google's Gemini AI model to answer statistical questions about the distributions. It's particularly useful for:

- Generating various types of data visualizations
- Creating training data for AI models
- Testing AI reasoning capabilities with statistical questions
- Analyzing text patterns and distributions
- Building datasets for visual question answering (VQA)

## Features

- Multiple visualization types:
  - Bar charts for letter frequency analysis
  - Pie charts for percentage-based analysis
  - Word clouds for corpus-level text analysis
- Text analysis with customizable frequency thresholds
- AI-powered question answering using Google's Gemini model
- Support for various statistical questions:
  - Letter/word frequency queries
  - Median frequency calculations
  - Mean frequency analysis
  - K-th most frequent element identification
  - Percentage-based analysis for pie charts
- Question rewriting capabilities for diverse training data
- Answer validation with tolerance for floating-point comparisons
- Dataset generation and upload capabilities to Hugging Face Hub

## Prerequisites

- Python 3.x
- Google Gemini API key
- Hugging Face account (for dataset upload)
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd reasoning_charts
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your API keys:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

#### Bar Chart Analysis
```python
from sentence_level.generate_barplot_sample import analyze_text_with_barplot

# Example text analysis
sample_text = "This is a sample text with some repeated letters. Let's see how many times each letter appears!"
question, expected_answer, analysis = analyze_text_with_barplot(
    text=sample_text,
    frequency_threshold=3,
    output_filename="letter_frequency.png"
)

print(f"Question: {question}")
print(f"Expected Answer: {expected_answer}")
print(f"Gemini's Analysis: {analysis}")
```

#### Pie Chart Analysis
```python
from sentence_level.generate_piechart_sample import analyze_text_with_piechart

# Example text analysis
sample_text = "This is a sample text with some repeated letters. Let's see how many times each letter appears!"
question, expected_answer, answer_type, analysis = analyze_text_with_piechart(
    text=sample_text,
    frequency_threshold=3,
    output_filename="letter_frequency_pie.png"
)
```

#### Word Cloud Analysis
```python
from corpus_level.generate_wordcloud_sample import analyze_text_with_wordcloud

# Example corpus analysis
sample_texts = [
    "First sample text with some words",
    "Second sample text with different words",
    "Third sample text showing word frequencies"
]
question, expected_answer, analysis = analyze_text_with_wordcloud(
    texts=sample_texts,
    frequency_threshold=2,
    output_filename="wordcloud.png"
)
```

### Generating Training Data

The project includes functionality to generate training data for AI models:

```python
from generate_train_set import main

# Generate training data with different visualization types
main(sentence_level=True, corpus_level=True, dataset_name="gsm8k", dataset_split="main")
```

### Uploading to Hugging Face

To upload generated datasets to Hugging Face Hub:

```python
from upload_created_dataset import main

# Upload dataset to Hugging Face
main(input_path="train_set_v3.json", repo_name="your-username/dataset-name")
```

## Project Structure

```
reasoning_charts/
├── sentence_level/
│   ├── generate_barplot_sample.py    # Bar chart generation and analysis
│   ├── generate_piechart_sample.py   # Pie chart generation and analysis
│   ├── sentence_to_barplot.py        # Bar chart utilities
│   └── sentence_to_piechart.py       # Pie chart utilities
├── corpus_level/
│   ├── generate_wordcloud_sample.py  # Word cloud generation and analysis
│   └── corpus_to_wordcloud.py        # Word cloud utilities
├── saved_graphs/                     # Directory for generated charts
├── train_set_v3.json                # Training data
├── generate_train_set.py            # Training data generation script
├── upload_created_dataset.py        # Dataset upload script
├── requirements.txt                 # Project dependencies
└── README.md                       # This file
```

## Answer Format

The system expects answers in LaTeX-style boxed format:
- Single backslash: `$\boxed{answer}$`
- Double backslash: `$\\boxed{answer}$`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT.