"""
segmentation.py - Implements greedy sentence chunking for long documents.
"""

import re
from typing import List

def simple_sentence_splitter(text: str) -> List[str]:
    """
    A simple regex-based sentence splitter.
    Splits text based on ., !, ? followed by a space or newline.
    This is a fallback to avoid NLTK dependency issues.
    """
    if not text:
        return []
    # Split by sentence-ending punctuation followed by whitespace or end of string.
    # The regex keeps the delimiters.
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def greedy_sentence_chunking(text: str, model_tokenizer, max_length: int) -> List[str]:
    """
    Splits a long document into shorter, non-overlapping segments using greedy sentence chunking.

    This function recursively adds sentences to a chunk until the number of tokens
    in the chunk exceeds the specified maximum length. This strategy helps preserve
    the syntax and semantics of sentences.

    Args:
        text: The long document to be segmented.
        model_tokenizer: The tokenizer from the pre-trained model (e.g., BERT tokenizer).
        max_length: The maximum number of tokens allowed per segment (e.g., 32, 64, 128).

    Returns:
        A list of text segments (chunks).
    """
    if not text:
        return []

    # 1. Split the document into individual sentences using the simple splitter
    sentences = simple_sentence_splitter(text)

    # 2. Greedily create chunks
    chunks = []
    current_chunk_sentences = []
    current_chunk_length = 0

    for sentence in sentences:
        # Tokenize the sentence and check its length
        sentence_tokens = model_tokenizer.tokenize(sentence)
        sentence_length = len(sentence_tokens)

        # If adding the new sentence exceeds the max length, finalize the current chunk
        if current_chunk_length + sentence_length > max_length and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            # Start a new chunk
            current_chunk_sentences = []
            current_chunk_length = 0
        
        # Add the new sentence to the current chunk
        current_chunk_sentences.append(sentence)
        current_chunk_length += sentence_length

    # 3. Add the last remaining chunk if it exists
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks 