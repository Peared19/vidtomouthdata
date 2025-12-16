import csv
import json
import os

INPUT_CSV = "gridcorpus/mouth_data_phoneme.csv"
OUTPUT_JSON = "phonemes.json"

def generate_phoneme_vocab():
    print(f"Reading {INPUT_CSV}...")
    
    unique_phonemes = set()
    
    # Special tokens
    # 0 is usually reserved for padding in many frameworks, so we'll start IDs from 1 or use 0 for a special token.
    # Let's define our special tokens first.
    special_tokens = ["<PAD>", "<START>", "<END>", "sil", "sp"]
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    try:
        with open(INPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                ph = row['curr_phoneme']
                if ph not in special_tokens:
                    unique_phonemes.add(ph)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Sort phonemes for consistency
    sorted_phonemes = sorted(list(unique_phonemes))
    
    # Create mappings
    phoneme_to_id = {token: idx for idx, token in enumerate(special_tokens)}
    
    start_idx = len(special_tokens)
    for idx, ph in enumerate(sorted_phonemes):
        phoneme_to_id[ph] = start_idx + idx
        
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    
    vocab = {
        "phoneme_to_id": phoneme_to_id,
        "id_to_phoneme": id_to_phoneme
    }
    
    print(f"Found {len(sorted_phonemes)} unique phonemes (excluding special tokens).")
    print(f"Total vocabulary size: {len(phoneme_to_id)}")
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4)
        
    print(f"Saved vocabulary to {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_phoneme_vocab()
