"""
Sentence splitting utilities for streaming text processing.
"""
import re
from typing import List, Tuple


def detect_complete_sentences(text: str, previous_buffer: str = "") -> Tuple[List[str], str]:
    """
    Detect complete sentences in text using regex.
    Optimized for streaming text with minimal latency.
    
    Args:
        text: New text to process
        previous_buffer: Previously buffered incomplete text
        
    Returns:
        Tuple of (list of complete sentences, remaining buffer)
    """
    # Combine previous buffer with new text
    combined = previous_buffer + text
    
    # Pattern to match sentence endings: . ! ? followed by space and capital letter, or end of string
    # This is more aggressive to catch sentences early
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\s*$)'
    
    # Split on sentence boundaries
    parts = re.split(sentence_pattern, combined)
    
    complete_sentences = []
    buffer = ""
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
            
        # Check if this part ends with sentence punctuation
        if re.search(r'[.!?]$', part):
            # This is a complete sentence
            complete_sentences.append(part)
        else:
            # This might be incomplete
            if i == len(parts) - 1:
                # Last part - keep in buffer for next chunk
                buffer = part
            else:
                # Middle part without punctuation - might be a fragment
                # If it's substantial, treat as sentence (for streaming)
                if len(part) > 10:  # Heuristic: substantial text without punctuation
                    complete_sentences.append(part)
                else:
                    buffer = part
    
    return complete_sentences, buffer


def split_on_sentences(text: str) -> List[str]:
    """
    Split text into sentences. Simple version for non-streaming use.
    """
    # Pattern to match sentence endings
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    # Filter out empty strings and strip
    return [s.strip() for s in sentences if s.strip()]

