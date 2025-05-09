from PIL import Image
from google import genai
import os
from dotenv import load_dotenv
from sentence_level.sentence_to_barplot import SentenceToBarplot
import random

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

def analyze_text_with_graph(text: str, frequency_threshold: int = 3, output_filename: str = "letter_frequency.png") -> tuple[str, str, str]:
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
    
    # Get Gemini's analysis
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17",
        contents=[image, question]
    )
    
    return question, expected_answer, response.text

if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample text with some repeated letters. Let's see how many times each letter appears!"
    question, expected_answer, analysis = analyze_text_with_graph(sample_text, frequency_threshold=3)
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Gemini's Analysis: {analysis}")

