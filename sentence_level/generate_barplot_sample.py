from PIL import Image
from google import genai
import os
from dotenv import load_dotenv
from sentence_level.sentence_to_barplot import SentenceToBarplot
import random
import re

load_dotenv()

def get_random_question(graph_generator: SentenceToBarplot) -> tuple[str, str]:
    """
    Generate a random question based on the statistical functions available.
    
    Args:
        graph_generator (SentenceToGraph): Instance of SentenceToGraph to get valid ranges
        
    Returns:
        tuple[str, str]: A tuple containing (question, expected_answer)
    """
    # Get the number of frequent letters to ensure k is valid
    num_letters = graph_generator.get_frequent_letters_count()
    random_value = random.randint(1, num_letters)   
    
    # Define possible questions and their corresponding function calls
    questions = [
        (
            "How many different letters appear in this frequency distribution?",
            f"Number of frequent letters: {num_letters}"
        ),
        (
            f"What is the frequency of the {random_value}th most frequent letter?",
            f"K-th frequency: {graph_generator.get_kth_frequency(random_value-1)}"
        ),
        (
            "What is the average frequency of letters in this distribution?",
            f"Mean frequency: {graph_generator.get_mean_frequency():.2f}"
        ),
        (
            "What is the median frequency of letters in this distribution?",
            f"Median frequency: {graph_generator.get_median_frequency():.2f}"
        )
    ]
    
    return random.choice(questions)

def rewrite_question_with_gemini(client: genai.Client, question: str) -> str:
    """
    Use Gemini to rewrite the question while maintaining its meaning.
    
    Args:
        client (genai.Client): Initialized Gemini client
        question (str): Original question to rewrite
        
    Returns:
        str: Rewritten question
    """
    prompt = f"Rewrite this question in a different way while keeping the same meaning: {question}. Output the rewritten question and only the rewritten question."
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17",
        contents=[prompt]
    )
    return response.text

def check_barplot_answer(expected_answer: str, model_answer: str) -> bool:
    """
    Check if the model's answer matches the expected answer using regex.
    
    Args:
        expected_answer (str): The expected answer string
        model_answer (str): The model's response
        
    Returns:
        bool: True if the answer matches, False otherwise
    """
    # Extract the answer from within \boxed{} if present, handling both single and double backslash formats
    answer_match = re.search(r'\$\\?\\boxed{(.*?)}\$', model_answer, re.DOTALL)
    if answer_match:
        model_answer = answer_match.group(1).strip()
    
    # Extract numeric values from both strings
    expected_numbers = re.findall(r'\d+\.?\d*', expected_answer)
    model_numbers = re.findall(r'\d+\.?\d*', model_answer)
    
    # If no numbers found, do a simple string comparison
    if not expected_numbers:
        return expected_answer.lower() in model_answer.lower()
    
    # Compare numeric values
    if len(expected_numbers) != len(model_numbers):
        return False
    
    # Compare each number with some tolerance for floating point values
    for exp_num, mod_num in zip(expected_numbers, model_numbers):
        try:
            exp_float = float(exp_num)
            mod_float = float(mod_num)
            if abs(exp_float - mod_float) > 0.01:  # 0.01 tolerance for floating point comparison
                return False
        except ValueError:
            if exp_num != mod_num:
                return False
    
    return True

def analyze_text_with_barplot(text: str, frequency_threshold: int = 3, output_filename: str = "letter_frequency.png") -> tuple[str, str, str]:
    """
    Analyze text by generating a frequency graph and using Gemini to analyze it.
    
    Args:
        text (str): Input text to analyze
        frequency_threshold (int): Minimum frequency for letters to be included in the graph
        output_filename (str): Name of the file to save the graph
        
    Returns:
        tuple[str, str, str]: A tuple containing (question, expected_answer, gemini_analysis)
    """
    # Initialize the graph generator
    graph_generator = SentenceToBarplot()
    
    # Create frequent letters dictionary
    frequent_letters = graph_generator.create_frequent_letters_dict(text, frequency_threshold)
    
    # Generate the graph
    graph_generator.generate_frequency_graph(
        frequent_letters=frequent_letters,
        filename=output_filename
    )
    
    # Get random question and expected answer
    question, expected_answer = get_random_question(graph_generator)
    
    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    # Rewrite question with 2/3 probability
    if random.random() < 2/3:
        question = rewrite_question_with_gemini(client, question)
    
    # Load the generated image
    image = Image.open(output_filename)
    
    # Get Gemini's analysis with modified prompt
    prompt = f"""{question}\n""" + r"Now think and answer the question. Put your answer within $\boxed{}$ tags."
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17",
        contents=[image, prompt]
    )
    
    return question, expected_answer, response.text

if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample text with some repeated letters. Let's see how many times each letter appears!"
    question, expected_answer, analysis = analyze_text_with_barplot(sample_text, frequency_threshold=3)
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Gemini's Analysis: {analysis}")
    print(f"Answer is correct: {check_barplot_answer(expected_answer, analysis)}")

