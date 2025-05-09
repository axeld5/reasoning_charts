import matplotlib.pyplot as plt
from collections import Counter
import random
from typing import Dict, Optional, Set
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords

class CorpusToWordcloud:
    def __init__(self):
        """
        Initialize the CorpusToWordcloud class.
        This class provides functionality to generate word clouds from text corpora.
        """
        self._word_frequencies: Optional[Dict[str, int]] = None
        self._all_words: Optional[Set[str]] = None
        # List of colormaps that work well for word clouds
        self._colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                          'cool', 'rainbow', 'jet', 'tab10', 'Set3', 'Paired']
        # Word cloud style variations
        self._background_colors = ['white', 'black', '#f0f0f0', '#e6e6e6']
        self._max_words = [100, 150, 200]
        self._widths = [800, 1000, 1200]
        self._heights = [400, 500, 600]

    def _get_random_colors(self, n_colors: int) -> list:
        """
        Generate random colors using a random colormap.
        
        Args:
            n_colors (int): Number of colors to generate
            
        Returns:
            list: List of color values
        """
        cmap_name = random.choice(self._colormaps)
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i/n_colors) for i in range(n_colors)]

    def generate_frequency_wordcloud(self, texts: list[str], frequency_threshold: int = 3) -> Dict[str, int]:
        """
        Create a dictionary of words that appear more than the threshold times across multiple texts.
        Stopwords and numbers are excluded from the frequency count.
        
        Args:
            texts (list[str]): List of input texts to analyze
            frequency_threshold (int): Minimum frequency for a word to be included
            
        Returns:
            Dict[str, int]: Dictionary of frequent words and their counts
        """
        # Initialize counters and sets
        all_words = set()
        word_counts = Counter()
        stop_words = set(stopwords.words('english'))
        
        # Process each text
        for text in texts:
            # Convert text to lowercase and split into words
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Filter out stopwords and numbers
            words = [word for word in words if word not in stop_words and not word.isdigit()]
            
            # Update word counts and unique words set
            word_counts.update(words)
            all_words.update(words)
        
        # Store all unique words for random word selection
        self._all_words = all_words
        
        # Filter words that appear more than the threshold
        self._word_frequencies = {word: count for word, count in word_counts.items() 
                                if count > frequency_threshold}
        return self._word_frequencies

    def get_wordcloud_words_count(self) -> int:
        """
        Get the number of words in the wordcloud.
        
        Returns:
            int: Number of words in the wordcloud
        """
        if self._word_frequencies is None:
            raise ValueError("No word frequencies dictionary created yet. Call generate_frequency_wordcloud first.")
        return len(self._word_frequencies)

    def get_kth_frequent_word(self, k: int) -> tuple[str, int]:
        """
        Get the k-th most frequent word and its count.
        
        Args:
            k (int): Index of the frequency to get (0 for highest, -1 for lowest)
            
        Returns:
            tuple: (word, frequency)
            
        Raises:
            ValueError: If k is out of bounds or no frequent words exist
        """
        if self._word_frequencies is None:
            raise ValueError("No word frequencies dictionary created yet. Call generate_frequency_wordcloud first.")
        
        if not self._word_frequencies:
            raise ValueError("No frequent words found.")
            
        if k >= len(self._word_frequencies) or k < -len(self._word_frequencies):
            raise ValueError(f"k must be between {-len(self._word_frequencies)} and {len(self._word_frequencies)-1}")
            
        # Sort by frequency in descending order
        sorted_items = sorted(self._word_frequencies.items(), key=lambda x: x[1], reverse=True)
        word, frequency = sorted_items[k]
        
        return word, frequency

    def is_word_in_wordcloud(self, word: str) -> bool:
        """
        Check if a word is present in the wordcloud.
        
        Args:
            word (str): The word to check
            
        Returns:
            bool: True if the word is in the wordcloud, False otherwise
            
        Raises:
            ValueError: If no word frequencies exist
        """
        if self._word_frequencies is None:
            raise ValueError("No word frequencies dictionary created yet. Call generate_frequency_wordcloud first.")
            
        return word.lower() in self._word_frequencies

    def get_random_word(self) -> str:
        """
        Get a random word from the corpus.
        
        Returns:
            str: A random word from the corpus
            
        Raises:
            ValueError: If no words exist in the corpus
        """
        if self._all_words is None or not self._all_words:
            raise ValueError("No words in corpus. Call generate_frequency_wordcloud first.")
            
        return random.choice(list(self._all_words))

    def generate_wordcloud(self, 
                          word_frequencies: Dict[str, int],
                          filename: str = 'wordcloud.png', 
                          dpi: int = 300) -> bool:
        """
        Generate and save a word cloud from a dictionary of word frequencies.
        
        Args:
            word_frequencies (Dict[str, int]): Dictionary of words and their frequencies
            filename (str): Name of the file to save the word cloud (default: 'wordcloud.png')
            dpi (int): Resolution of the saved image (default: 300)
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            if not word_frequencies:
                print("No word frequencies provided.")
                return False
            
            # Randomly select word cloud style parameters
            background_color = random.choice(self._background_colors)
            max_words = random.choice(self._max_words)
            width = random.choice(self._widths)
            height = random.choice(self._heights)
            
            # Create word cloud with more pronounced size differences
            wordcloud = WordCloud(
                background_color=background_color,
                max_words=max_words,
                width=width,
                height=height,
                colormap=random.choice(self._colormaps),
                relative_scaling=1,  # Makes size differences more pronounced (0.0 to 1.0)
                min_font_size=8,       # Ensures smallest words are still readable
                max_font_size=150,      # Allows for larger maximum word size
                stopwords=set(stopwords.words('english'))
            ).generate_from_frequencies(word_frequencies)
            
            # Create the figure
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Frequency Distribution')
            
            # Save the figure
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Word cloud saved successfully as {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving word cloud: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Example texts
    sample_texts = [
        "Test test test word word thing thing thing thing a burp",
        "Another test text with different words and more complexity",
        "Final example showing how multiple texts combine their word frequencies",
        "e e e e e e e e e e e e"
    ]
    
    # Create instance
    analyzer = CorpusToWordcloud()
    
    # Create word frequencies dictionary from multiple texts
    word_frequencies = analyzer.generate_frequency_wordcloud(sample_texts, frequency_threshold=3)
    print(f"Word frequencies: {word_frequencies}")
    
    # Demonstrate statistical functions
    print(f"Number of words in wordcloud: {analyzer.get_wordcloud_words_count()}")
    
    # Get most frequent word
    word, freq = analyzer.get_kth_frequent_word(0)
    print(f"Most frequent word: {word} (frequency: {freq})")
    
    # Get least frequent word
    word, freq = analyzer.get_kth_frequent_word(-1)
    print(f"Least frequent word: {word} (frequency: {freq})")
    
    # Check if a random word is in the wordcloud
    random_word = analyzer.get_random_word()
    print(f"Is '{random_word}' in wordcloud? {analyzer.is_word_in_wordcloud(random_word)}")
    
    # Generate word clouds using the word frequencies dictionary
    analyzer.generate_wordcloud(
        word_frequencies=word_frequencies,
        filename='wordcloud1.png'
    )
    
    analyzer.generate_wordcloud(
        word_frequencies=word_frequencies,
        filename='wordcloud2.png'
    )
    
    analyzer.generate_wordcloud(
        word_frequencies=word_frequencies,
        filename='wordcloud3.png'
    ) 