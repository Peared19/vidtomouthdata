"""
Generate vocabulary.json from mouth_data_context.csv
Creates word-to-ID and ID-to-word mappings with special tokens
"""

import json
import pandas as pd
import argparse
from pathlib import Path


def generate_vocabulary(csv_path: str, output_path: str = 'vocabulary.json'):
    """
    Generate vocabulary from mouth_data_context.csv
    
    Args:
        csv_path: Path to mouth_data_context.csv
        output_path: Path to save vocabulary.json
    """
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, delimiter=';')
    print(f"Loaded {len(df)} samples")
    
    # Extract all unique words
    unique_words = set()
    
    # From curr_word (main content)
    unique_words.update(df['curr_word'].unique())
    
    # From prev_word and next_word (might have <START>/<END>)
    unique_words.update(df['prev_word'].unique())
    unique_words.update(df['next_word'].unique())
    
    # Remove any NaN or whitespace
    unique_words = {w for w in unique_words if isinstance(w, str) and w.strip()}
    
    print(f"Found {len(unique_words)} unique words")
    
    # Sort for consistency
    sorted_words = sorted(unique_words)
    print(f"Vocabulary: {sorted_words}")
    
    # Create word-to-ID mapping
    # Reserve 0 for <START>, last for <END>
    word_to_id = {'<START>': 0}
    
    # Map actual words starting from ID 1
    for idx, word in enumerate(sorted_words, start=1):
        word_to_id[word] = idx
    
    # <END> token gets the last ID
    word_to_id['<END>'] = len(word_to_id)
    
    # Create ID-to-word mapping (reverse)
    id_to_word = {v: k for k, v in word_to_id.items()}
    
    # Create final vocabulary structure
    vocabulary = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'vocab_size': len(word_to_id),
        'unique_words': len(unique_words),
        'special_tokens': ['<START>', '<END>']
    }
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    
    print(f"\nVocabulary saved to {output_path}")
    print(f"Total vocab size: {vocabulary['vocab_size']}")
    print(f"Unique words: {vocabulary['unique_words']}")
    print(f"Special tokens: {vocabulary['special_tokens']}")
    
    # Print some statistics
    print("\nWord-to-ID mapping (sample):")
    for i, (word, word_id) in enumerate(list(word_to_id.items())[:10]):
        print(f"  {word:20s} → {word_id}")
    print("  ...")
    
    return vocabulary


def validate_vocabulary(vocab_path: str, csv_path: str):
    """
    Validate that vocabulary covers all words in the CSV
    
    Args:
        vocab_path: Path to vocabulary.json
        csv_path: Path to mouth_data_context.csv
    """
    print(f"\nValidating vocabulary...")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    word_to_id = vocab['word_to_id']
    
    df = pd.read_csv(csv_path, delimiter=';')
    
    # Check all words in CSV
    all_words_in_csv = set()
    all_words_in_csv.update(df['curr_word'].unique())
    all_words_in_csv.update(df['prev_word'].unique())
    all_words_in_csv.update(df['next_word'].unique())
    
    # Remove NaN/whitespace
    all_words_in_csv = {w for w in all_words_in_csv if isinstance(w, str) and w.strip()}
    
    # Check coverage
    missing_words = all_words_in_csv - set(word_to_id.keys())
    
    if missing_words:
        print(f"WARNING: {len(missing_words)} words in CSV not in vocabulary:")
        for word in sorted(missing_words):
            print(f"  - {word}")
        return False
    else:
        print(f"✓ All {len(all_words_in_csv)} words in CSV are in vocabulary")
        return True


def main():
    parser = argparse.ArgumentParser(description='Generate vocabulary from mouth_data_context.csv')
    parser.add_argument('--csv', default='gridcorpus/mouth_data_context.csv',
                        help='Path to mouth_data_context.csv')
    parser.add_argument('--output', default='vocabulary.json',
                        help='Output path for vocabulary.json')
    parser.add_argument('--validate', action='store_true',
                        help='Validate vocabulary after generation')
    
    args = parser.parse_args()
    
    # Generate
    vocab = generate_vocabulary(args.csv, args.output)
    
    # Validate if requested
    if args.validate:
        is_valid = validate_vocabulary(args.output, args.csv)
        if is_valid:
            print("\n✓ Vocabulary validation passed!")
        else:
            print("\n✗ Vocabulary validation failed!")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
