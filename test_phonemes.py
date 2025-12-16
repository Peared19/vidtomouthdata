import json
from g2p_en import G2p
import nltk

def ensure_nltk_resources():
    """Ensure necessary NLTK data is downloaded."""
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading missing NLTK resource: averaged_perceptron_tagger_eng...")
        nltk.download('averaged_perceptron_tagger_eng')

def test_phoneme_conversion():
    ensure_nltk_resources()
    print("Initializing G2P (Grapheme-to-Phoneme)...")
    g2p = G2p()
    
    # Load our existing vocabulary to see how it translates
    try:
        with open('vocabulary.json', 'r') as f:
            vocab = json.load(f)
        words = [w for w in vocab['word_to_id'].keys() if w not in ['<START>', '<END>', 'sil', 'sp']]
    except FileNotFoundError:
        print("vocabulary.json not found, using sample words.")
        words = ["blue", "green", "place", "set", "zero"]

    print(f"\nConverting {len(words)} words to phonemes:\n")
    
    # Test a few specific words
    test_words = words[:5] + ["cyan", "magenta", "artificial"] # Add some unseen words
    
    for word in test_words:
        # g2p returns a list of phonemes
        phonemes = g2p(word)
        # Filter out non-phoneme characters (like spaces or numbers if any, though g2p_en usually handles this)
        phonemes = [p for p in phonemes if p != ' ']
        
        print(f"Word: {word:15} -> Phonemes: {phonemes}")

    print("\n------------------------------------------------")
    print("Explanation:")
    print("These symbols (e.g., 'B', 'L', 'UW1') are ARPABET phonemes.")
    print("- 'UW1' means the 'oo' sound in 'blue' with primary stress.")
    print("- 'S' is the 's' sound.")
    print("- 'AY1' is the 'eye' sound.")
    print("------------------------------------------------")

if __name__ == "__main__":
    test_phoneme_conversion()
