# Reasoning Charts

A Python project that generates and analyzes letter frequency distributions from text, with AI-powered question answering capabilities.

## Overview

This project analyzes text input to create bar charts of letter frequencies and uses Google's Gemini AI model to answer statistical questions about the distributions. It's particularly useful for:

- Generating letter frequency visualizations
- Creating training data for AI models
- Testing AI reasoning capabilities with statistical questions
- Analyzing text patterns and distributions

## Features

- Text analysis with customizable frequency thresholds
- Automatic bar chart generation
- AI-powered question answering using Google's Gemini model
- Support for various statistical questions:
  - Letter frequency queries
  - Median frequency calculations
  - Mean frequency analysis
  - K-th most frequent letter identification
- Question rewriting capabilities for diverse training data
- Answer validation with tolerance for floating-point comparisons

## Prerequisites

- Python 3.x
- Google Gemini API key
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

3. Create a `.env` file in the root directory and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from sentence_level.generate_sample import analyze_text_with_graph

# Example text analysis
sample_text = "This is a sample text with some repeated letters. Let's see how many times each letter appears!"
question, expected_answer, analysis = analyze_text_with_graph(
    text=sample_text,
    frequency_threshold=3,
    output_filename="letter_frequency.png"
)

print(f"Question: {question}")
print(f"Expected Answer: {expected_answer}")
print(f"Gemini's Analysis: {analysis}")
```

### Generating Training Data

The project includes functionality to generate training data for AI models:

```python
from sentence_level.generate_sample import analyze_text_with_graph

# Generate multiple samples
texts = [
    "Sample text 1",
    "Sample text 2",
    # Add more texts as needed
]

for i, text in enumerate(texts):
    question, expected_answer, analysis = analyze_text_with_graph(
        text=text,
        frequency_threshold=3,
        output_filename=f"saved_graphs/graph_{i}.png"
    )
```

## Project Structure

```
reasoning_charts/
├── sentence_level/
│   ├── generate_sample.py    # Main analysis and generation logic
│   └── sentence_to_barplot.py # Bar chart generation utilities
├── saved_graphs/             # Directory for generated charts
├── train_set.json           # Training data
├── generate_train_set.py    # Training data generation script
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Answer Format

The system expects answers in LaTeX-style boxed format:
- Single backslash: `$\boxed{answer}$`
- Double backslash: `$\\boxed{answer}$`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT.