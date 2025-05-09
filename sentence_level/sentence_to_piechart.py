import matplotlib.pyplot as plt
from collections import Counter
import random
from typing import Dict, Optional

class SentenceToPiechart:
    def __init__(self):
        """
        Initialize the SentenceToPiechart class.
        This class provides functionality to generate frequency piecharts from text.
        """
        self._frequent_letters: Optional[Dict[str, int]] = None
        # List of colormaps that work well for pie charts
        self._colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                          'cool', 'rainbow', 'jet', 'tab10', 'Set3', 'Paired']
        # Pie chart style variations
        self._explode_values = [0.05, 0.1, 0.15]  # Explode values for pie slices
        self._shadow_options = [True, False]
        self._start_angles = [0, 90, 180, 270]
        self._wedge_props = [{'linewidth': 1, 'edgecolor': 'white'},
                            {'linewidth': 2, 'edgecolor': 'black'},
                            {'linewidth': 1.5, 'edgecolor': 'gray'}]

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

    def create_frequent_letters_dict(self, text: str, frequency_threshold: int = 3) -> Dict[str, int]:
        """
        Create a dictionary of letters that appear more than the threshold times.
        
        Args:
            text (str): The input text to analyze
            frequency_threshold (int): Minimum frequency for a letter to be included
            
        Returns:
            Dict[str, int]: Dictionary of frequent letters and their counts
        """
        # Count letter frequencies
        letter_counts = Counter(char for char in text if char.isalpha())
        
        # Filter letters that appear more than the threshold
        self._frequent_letters = {letter: count for letter, count in letter_counts.items() 
                                if count > frequency_threshold}
        
        return self._frequent_letters

    def get_frequent_letters_count(self) -> int:
        """
        Get the number of frequent letters.
        
        Returns:
            int: Number of frequent letters
        """
        if self._frequent_letters is None:
            raise ValueError("No frequent letters dictionary created yet. Call create_frequent_letters_dict first.")
        return len(self._frequent_letters)

    def get_kth_frequent_element(self, k: int) -> tuple[str, int, float]:
        """
        Get the k-th most frequent element and its percentage.
        
        Args:
            k (int): Index of the frequency to get (0 for highest, -1 for lowest)
            
        Returns:
            tuple: (letter, frequency, percentage)
            
        Raises:
            ValueError: If k is out of bounds or no frequent letters exist
        """
        if self._frequent_letters is None:
            raise ValueError("No frequent letters dictionary created yet. Call create_frequent_letters_dict first.")
        
        if not self._frequent_letters:
            raise ValueError("No frequent letters found.")
            
        if k >= len(self._frequent_letters) or k < -len(self._frequent_letters):
            raise ValueError(f"k must be between {-len(self._frequent_letters)} and {len(self._frequent_letters)-1}")
            
        # Sort by frequency in descending order
        sorted_items = sorted(self._frequent_letters.items(), key=lambda x: x[1], reverse=True)
        letter, frequency = sorted_items[k]
        
        # Calculate percentage
        total = sum(self._frequent_letters.values())
        percentage = (frequency / total) * 100
        
        return letter, frequency, percentage

    def generate_frequency_graph(self, 
                               frequent_letters: Dict[str, int],
                               filename: str = 'letter_frequency_pie.png', 
                               dpi: int = 300) -> bool:
        """
        Generate and save a frequency pie chart from a dictionary of letter frequencies.
        
        Args:
            frequent_letters (Dict[str, int]): Dictionary of letters and their frequencies
            filename (str): Name of the file to save the graph (default: 'letter_frequency_pie.png')
            dpi (int): Resolution of the saved image (default: 300)
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            if not frequent_letters:
                print("No frequent letters provided.")
                return False
            
            # Create the figure
            plt.figure(figsize=(10, 8))
            
            # Generate random colors
            colors = self._get_random_colors(len(frequent_letters))
            
            # Randomly select pie chart style parameters
            explode = [random.choice(self._explode_values) for _ in range(len(frequent_letters))]
            shadow = random.choice(self._shadow_options)
            start_angle = random.choice(self._start_angles)
            wedge_props = random.choice(self._wedge_props)
            
            # Calculate percentages for labels
            total = sum(frequent_letters.values())
            labels = [f'{letter} ({count}, {count/total*100:.1f}%)' 
                     for letter, count in frequent_letters.items()]
            
            # Create pie chart
            plt.pie(frequent_letters.values(),
                   labels=labels,
                   colors=colors,
                   explode=explode,
                   shadow=shadow,
                   startangle=start_angle,
                   wedgeprops=wedge_props,
                   autopct='%1.1f%%')
            
            plt.title('Letter Frequency Distribution')
            
            # Save the figure
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Pie chart saved successfully as {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving pie chart: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Example text
    sample_text = "This is a sample text with some repeated letters. Let's see how many times each letter appears!"
    
    # Create instance
    analyzer = SentenceToPiechart()
    
    # Create frequent letters dictionary
    frequent_letters = analyzer.create_frequent_letters_dict(sample_text, frequency_threshold=3)
    print(f"Frequent letters: {frequent_letters}")
    
    # Demonstrate statistical functions
    print(f"Number of frequent letters: {analyzer.get_frequent_letters_count()}")
    
    # Get most frequent element
    letter, freq, pct = analyzer.get_kth_frequent_element(0)
    print(f"Most frequent letter: {letter} (frequency: {freq}, {pct:.1f}%)")
    
    # Get least frequent element
    letter, freq, pct = analyzer.get_kth_frequent_element(-1)
    print(f"Least frequent letter: {letter} (frequency: {freq}, {pct:.1f}%)")
    
    # Generate pie charts using the frequent letters dictionary
    analyzer.generate_frequency_graph(
        frequent_letters=frequent_letters,
        filename='letter_frequency_pie1.png'
    )
    
    analyzer.generate_frequency_graph(
        frequent_letters=frequent_letters,
        filename='letter_frequency_pie2.png'
    )
    
    analyzer.generate_frequency_graph(
        frequent_letters=frequent_letters,
        filename='letter_frequency_pie3.png'
    ) 