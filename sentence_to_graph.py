import matplotlib.pyplot as plt
from collections import Counter
import random
from typing import Dict, Optional
import statistics
import matplotlib.pyplot as plt

class SentenceToGraph:
    def __init__(self):
        """
        Initialize the SentenceToGraph class.
        This class provides functionality to generate frequency graphs from text.
        """
        self._frequent_letters: Optional[Dict[str, int]] = None
        # List of colormaps that work well for both pie and bar charts
        self._colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                          'cool', 'rainbow', 'jet', 'tab10', 'Set3', 'Paired']

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

    def get_kth_frequency(self, k: int) -> int:
        """
        Get the k-th frequency value from the sorted frequencies.
        
        Args:
            k (int): Index of the frequency to get (0 for highest, -1 for lowest)
            
        Returns:
            int: The k-th frequency value
            
        Raises:
            ValueError: If k is out of bounds or no frequent letters exist
        """
        if self._frequent_letters is None:
            raise ValueError("No frequent letters dictionary created yet. Call create_frequent_letters_dict first.")
        
        if not self._frequent_letters:
            raise ValueError("No frequent letters found.")
            
        if k >= len(self._frequent_letters) or k < -len(self._frequent_letters):
            raise ValueError(f"k must be between {-len(self._frequent_letters)} and {len(self._frequent_letters)-1}")
            
        # Sort frequencies in descending order
        sorted_frequencies = sorted(self._frequent_letters.values(), reverse=True)
        return sorted_frequencies[k]

    def get_mean_frequency(self) -> float:
        """
        Calculate the mean frequency of the frequent letters.
        
        Returns:
            float: Mean frequency
            
        Raises:
            ValueError: If no frequent letters exist
        """
        if self._frequent_letters is None:
            raise ValueError("No frequent letters dictionary created yet. Call create_frequent_letters_dict first.")
            
        if not self._frequent_letters:
            raise ValueError("No frequent letters found.")
            
        return statistics.mean(self._frequent_letters.values())

    def get_median_frequency(self) -> float:
        """
        Calculate the median frequency of the frequent letters.
        
        Returns:
            float: Median frequency
            
        Raises:
            ValueError: If no frequent letters exist
        """
        if self._frequent_letters is None:
            raise ValueError("No frequent letters dictionary created yet. Call create_frequent_letters_dict first.")
            
        if not self._frequent_letters:
            raise ValueError("No frequent letters found.")
            
        return statistics.median(self._frequent_letters.values())

    def generate_frequency_graph(self, 
                               frequent_letters: Dict[str, int],
                               filename: str = 'letter_frequency.png', 
                               dpi: int = 300,
                               plot_type: str = None) -> bool:
        """
        Generate and save a frequency graph from a dictionary of letter frequencies.
        
        Args:
            frequent_letters (Dict[str, int]): Dictionary of letters and their frequencies
            filename (str): Name of the file to save the graph (default: 'letter_frequency.png')
            dpi (int): Resolution of the saved image (default: 300)
            plot_type (str): Type of plot to generate ('pie' or 'bar'). If None, randomly chooses between them.
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            if not frequent_letters:
                print("No frequent letters provided.")
                return False
            
            # Create the figure
            plt.figure(figsize=(10, 6))
            
            # Generate random colors
            colors = self._get_random_colors(len(frequent_letters))
            
            # Randomly choose plot type if not specified
            if plot_type is None:
                plot_type = random.choice(['pie', 'bar'])
            
            if plot_type == 'pie':
                # Create pie chart
                plt.pie(frequent_letters.values(), 
                       labels=frequent_letters.keys(),
                       autopct=lambda pct: f'{int(pct * sum(frequent_letters.values()) / 100)}',
                       startangle=90,
                       colors=colors)
                plt.title(f'Letter Frequency Distribution (Pie Chart)')
            else:
                # Create bar plot
                bars = plt.bar(frequent_letters.keys(), 
                             frequent_letters.values(),
                             color=colors)
                plt.title(f'Letter Frequency Distribution (Bar Plot)')
                plt.xlabel('Letters')
                plt.ylabel('Frequency')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels on top of each bar
                for i, v in enumerate(frequent_letters.values()):
                    plt.text(i, v, str(v), ha='center', va='bottom')
            
            # Save the figure
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Graph saved successfully as {filename} ({plot_type} chart)")
            return True
            
        except Exception as e:
            print(f"Error saving graph: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Example text
    sample_text = "This is a sample text with some repeated letters. Let's see how many times each letter appears!"
    
    # Create instance
    analyzer = SentenceToGraph()
    
    # Create frequent letters dictionary
    frequent_letters = analyzer.create_frequent_letters_dict(sample_text, frequency_threshold=3)
    print(f"Frequent letters: {frequent_letters}")
    
    # Demonstrate statistical functions
    print(f"Number of frequent letters: {analyzer.get_frequent_letters_count()}")
    print(f"Highest frequency: {analyzer.get_kth_frequency(0)}")
    print(f"Lowest frequency: {analyzer.get_kth_frequency(-1)}")
    print(f"Mean frequency: {analyzer.get_mean_frequency():.2f}")
    print(f"Median frequency: {analyzer.get_median_frequency():.2f}")
    
    # Generate graphs using the frequent letters dictionary
    analyzer.generate_frequency_graph(
        frequent_letters=frequent_letters,
        filename='letter_frequency_random1.png'
    )
    
    analyzer.generate_frequency_graph(
        frequent_letters=frequent_letters,
        filename='letter_frequency_pie.png',
        plot_type='pie'
    )
    
    analyzer.generate_frequency_graph(
        frequent_letters=frequent_letters,
        filename='letter_frequency_bar.png',
        plot_type='bar'
    ) 