from PIL import Image
from google import genai
import os
from dotenv import load_dotenv
from sentence_level.sentence_to_piechart import SentenceToPiechart
import random
import re

load_dotenv()

def get_random_question(graph_generator: SentenceToPiechart) -> tuple[str, str, str]:
    """
    Generate a random question based on the statistical functions available.
    
    Args:
        graph_generator (SentenceToPiechart): Instance of SentenceToPiechart to get valid ranges
        
    Returns:
        tuple[str, str, str]: A tuple containing (question, expected_answer, answer_type)
    """
    # Get the number of frequent letters to ensure k is valid
    num_letters = graph_generator.get_frequent_letters_count()
    random_value = random.randint(1, num_letters)   
    
    # Define possible questions and their corresponding function calls
    questions = [
        (
            "How many different letters appear in this frequency distribution?",
            f"Number of frequent letters: {num_letters}",
            "count"
        ),
        (
            f"What letter has the {random_value}th highest frequency?",
            f"Letter: {graph_generator.get_kth_frequent_element(random_value-1)[0]}",
            "letter"
        ),
        (
            f"What is the percentage of the {random_value}th most frequent letter?",
            f"Percentage: {graph_generator.get_kth_frequent_element(random_value-1)[2]:.1f}%",
            "percentage"
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

def check_piechart_answer(expected_answer: str, model_answer: str, answer_type: str) -> bool:
    """
    Check if the model's answer matches the expected answer based on the answer type.
    
    Args:
        expected_answer (str): The expected answer string
        model_answer (str): The model's response
        answer_type (str): Type of answer to check ('count', 'letter', 'percentage', 'letter_percentage')
        
    Returns:
        bool: True if the answer matches, False otherwise
    """
    # Extract the answer from within \boxed{} if present
    answer_match = re.search(r'\$\\?\\boxed{(.*?)}\$', model_answer, re.DOTALL)
    if answer_match:
        model_answer = answer_match.group(1).strip()
    
    if answer_type == "count":
        # Extract numbers and compare
        expected_num = re.search(r'\d+', expected_answer)
        model_num = re.search(r'\d+', model_answer)
        if not expected_num or not model_num:
            return False
        return int(expected_num.group()) == int(model_num.group())
    
    elif answer_type == "letter":
        # Extract letters and compare
        expected_letter = re.search(r'Letter: ([A-Za-z])', expected_answer)
        model_letter = re.search(r'([A-Za-z])', model_answer)
        if not expected_letter or not model_letter:
            return False
        return expected_letter.group(1).lower() == model_letter.group(1).lower()
    
    elif answer_type == "percentage":
        # Extract percentages and compare with tolerance
        expected_pct = re.search(r'(\d+\.?\d*)\%?', expected_answer)
        model_pct = re.search(r'(\d+\.?\d*)\%?', model_answer)
        if not expected_pct or not model_pct:
            return False
        return abs(float(expected_pct.group(1)) - float(model_pct.group(1))) < 0.1
    
    return False

def analyze_text_with_piechart(text: str, frequency_threshold: int = 3, output_filename: str = "letter_frequency_pie.png") -> tuple[str, str, str, str]:
    """
    Analyze text by generating a frequency pie chart and using Gemini to analyze it.
    
    Args:
        text (str): Input text to analyze
        frequency_threshold (int): Minimum frequency for letters to be included in the graph
        output_filename (str): Name of the file to save the graph
        
    Returns:
        tuple[str, str, str, str]: A tuple containing (question, expected_answer, answer_type, gemini_analysis)
    """
    # Initialize the graph generator
    graph_generator = SentenceToPiechart()
    
    # Create frequent letters dictionary
    frequent_letters = graph_generator.create_frequent_letters_dict(text, frequency_threshold)
    
    # Generate the graph
    graph_generator.generate_frequency_graph(
        frequent_letters=frequent_letters,
        filename=output_filename
    )
    
    # Get random question and expected answer
    question, expected_answer, answer_type = get_random_question(graph_generator)
    
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
    
    return question, expected_answer, answer_type, response.text

if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample text with some repeated letters. Let's see how many times each letter appears!"
    question, expected_answer, answer_type, analysis = analyze_text_with_piechart(sample_text, frequency_threshold=3)
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Answer Type: {answer_type}")
    print(f"Gemini's Analysis: {analysis}")
    print(f"Answer is correct: {check_piechart_answer(expected_answer, analysis, answer_type)}") 