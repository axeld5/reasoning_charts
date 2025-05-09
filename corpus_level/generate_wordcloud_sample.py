from PIL import Image
from google import genai
import os
from dotenv import load_dotenv
from corpus_level.corpus_to_wordcloud import CorpusToWordcloud
import random
import re

load_dotenv()

def get_random_question(graph_generator: CorpusToWordcloud) -> tuple[str, str]:
    """
    Generate a random question based on the statistical functions available.
    
    Args:
        graph_generator (CorpusToWordcloud): Instance of CorpusToWordcloud to get valid ranges
        
    Returns:
        tuple[str, str]: A tuple containing (question, expected_answer)
    """
    # Get the number of words in wordcloud to ensure k is valid
    num_words = graph_generator.get_wordcloud_words_count()
    random_value = random.randint(1, min(3, num_words))
    
    # Get a random word from the corpus
    random_word = graph_generator.get_random_word()
    
    # Define possible questions and their corresponding function calls
    questions = [
        (
            "How many different words appear in this word cloud?",
            f"Number of words: {num_words}"
        ),
        (
            f"What is the {random_value}th most frequent word in this word cloud?",
            f"Word: {graph_generator.get_kth_frequent_word(random_value-1)[0]}"
        ),
        (
            f"Is the word '{random_word}' present in this word cloud?",
            f"Word present: {graph_generator.is_word_in_wordcloud(random_word)}"
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

def check_wordcloud_answer(expected_answer: str, model_answer: str) -> bool:
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
        return model_answer.lower().strip() in expected_answer.lower().strip()
    
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

def analyze_text_with_wordcloud(texts: list[str], frequency_threshold: int = 5, output_filename: str = "wordcloud.png") -> tuple[str, str, str]:
    """
    Analyze text by generating a word cloud and using Gemini to analyze it.
    
    Args:
        texts (list[str]): Input texts to analyze
        frequency_threshold (int): Minimum frequency for words to be included in the word cloud
        output_filename (str): Name of the file to save the word cloud
        
    Returns:
        tuple[str, str, str]: A tuple containing (question, expected_answer, gemini_analysis)
    """
    # Initialize the word cloud generator
    wordcloud_generator = CorpusToWordcloud()
    
    # Create word frequencies dictionary
    word_frequencies = wordcloud_generator.generate_frequency_wordcloud(texts, frequency_threshold)
    
    # Generate the word cloud
    wordcloud_generator.generate_wordcloud(
        word_frequencies=word_frequencies,
        filename=output_filename
    )
    
    # Get random question and expected answer
    question, expected_answer = get_random_question(wordcloud_generator)
    
    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    # Rewrite question with 2/3 probability
    if random.random() < 2/3:
        if "present" in question:
            question = rewrite_question_with_gemini(client, question)
            question += "\n Answer with True or False."
        else:
            question = rewrite_question_with_gemini(client, question)
    else:
        if "present" in question:
            question += "\n Answer with True or False."

    
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
    sample_texts = [
        "Test test test word word thing thing thing thing a burp",
        "Another test text with different words and more complexity",
        "Final example showing how multiple texts combine their word frequencies",
        "e e e e e e e e e e e e"
    ]
    question, expected_answer, analysis = analyze_text_with_wordcloud(sample_texts, frequency_threshold=2)
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Gemini's Analysis: {analysis}")
    print(f"Answer is correct: {check_wordcloud_answer(expected_answer, analysis)}") 