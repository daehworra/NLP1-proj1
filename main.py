from tokenizers import Tokenizer
from rep_spaces import TokenIndicator, BagOfWords
import numpy as np

if __name__ == "__main__": 
    # Pretrained tokenizer
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    np.set_printoptions(threshold=np.inf)  # Set threshold to infinity to print the entire array

    # Test the TokenIndicator
    token_indicator = TokenIndicator(tokenizer)
    test_phrase = "The quick quick brown fox jumps over the lazy dog"
    token_indicator_vector = token_indicator(test_phrase)
    print("TokenIndicator vector:", token_indicator_vector)

    #Test the BagOfWords
    bag_of_words = BagOfWords(tokenizer)
    bag_of_words_vector = bag_of_words(test_phrase)
    print("BagOfWords vector:", bag_of_words_vector)