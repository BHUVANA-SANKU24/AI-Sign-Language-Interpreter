"""
NLP Processing Module
Process text input and prepare for sign language generation
"""

import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)


class NLPProcessor:
    """Process text for sign language"""

    def __init__(self):
        """Initialize NLP processor"""
        print("✓ NLP processor initialized")

    def process_text(self, text):
        """
        Process text into tokens

        Args:
            text: Input text string

        Returns:
            tokens: List of word tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove punctuation
        tokens = [word for word in tokens if word.isalnum()]

        return tokens

    def text_to_sign_sequence(self, text, sign_vocabulary):
        """
        Convert text to sequence of signs

        Args:
            text: Input text
            sign_vocabulary: List/dict of available signs

        Returns:
            sign_sequence: List of signs to perform
        """
        tokens = self.process_text(text)
        sign_sequence = []

        # Convert dict to list if needed
        if isinstance(sign_vocabulary, dict):
            available_signs = list(sign_vocabulary.keys())
        else:
            available_signs = sign_vocabulary

        # Convert to uppercase for matching
        available_signs_upper = [s.upper() for s in available_signs]

        for token in tokens:
            token_upper = token.upper()

            # Check if word is in vocabulary
            if token_upper in available_signs_upper:
                sign_sequence.append(token_upper)
            else:
                # Fingerspell unknown words letter by letter
                for char in token:
                    if char.isalpha():
                        sign_sequence.append(char.upper())

        return sign_sequence

    def simplify_for_asl(self, text):
        """
        Simplify text for ASL grammar
        (ASL has different grammar than English)

        Args:
            text: Input text

        Returns:
            simplified: Simplified text
        """
        # Remove common English words that aren't typically signed
        stop_words = {'a', 'an', 'the', 'is', 'are', 'am', 'was', 'were'}

        tokens = self.process_text(text)
        filtered = [word for word in tokens if word not in stop_words]

        return ' '.join(filtered)
