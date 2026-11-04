import numpy as np
from tokenizers import Tokenizer

#TODO: Possibly exclude any unknown tokens from the tokeniser?

class TokenIndicator:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode(text).tokens
        vector = np.zeros(len(self), dtype=np.float32)

        for token in set(tokens):
            if token in self.vocab:
                vector[self.vocab[token]] = 1

        return vector


class BagOfWords:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode(text).tokens
        vector = np.zeros(len(self), dtype=np.float32)

        for token in tokens:
            if token in self.vocab:
                vector[self.vocab[token]] += 1

        return vector

# (Smoothed) Term Frequency-Inverse Document Frequency (TF-IDF) class
class TF_IDF:
    def __init__(self, tokenizer: Tokenizer, documents: list):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.document_count = len(documents)
        self.idf = self.compute_idf(documents)

    def compute_idf(self, documents):
        idf = np.zeros(len(self), dtype=np.float32)
        document_frequencies = np.zeros(len(self), dtype=np.float32)
        for document in documents:
            unique_tokens = set(self.tokenizer.encode(document).tokens)

            for token in unique_tokens:
                index = self.vocab[token]
                document_frequencies[index] += 1

        for token in self.vocab:
            index = self.vocab[token]
            idf[index] = np.log((self.document_count + 1) / (document_frequencies[index] + 1)) + 1

        return idf

    def compute_term_frequencies(self, document):
        term_frequencies = np.zeros(len(self), dtype=np.float32)
        tokens = self.tokenizer.encode(document).tokens

        for token in tokens:
            index = self.vocab[token]
            term_frequencies[index] +=1

        return term_frequencies


    def __call__(self, document):
        tf = self.compute_term_frequencies(document)
        return np.multiply(tf, self.idf)



            
