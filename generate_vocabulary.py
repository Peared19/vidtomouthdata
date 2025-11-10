#!/usr/bin/env python3
"""
Dataset után futtatandó: szóvocabulárium generálása
Használat: python generate_vocabulary.py
"""

import json
import csv
import sys

def generate_vocabulary(csv_path="D:/MestInt/datasets/gridcorpus/mouth_data.csv", 
                       output_path="vocabulary.json"):
    """
    CSV-ből kinyeri az összes unique szót és statisztikákat
    """
    
    try:
        words_list = []
        word_counts = {}
        speakers = set()
        total_frames = 0
        
        print(f"\n📖 Szóvocabulárium generálása...")
        print(f"   CSV: {csv_path}")
        
        # CSV betöltése és feldolgozása
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames is None:
                print(f"❌ CSV header hiányzik!")
                return None
            
            for row in reader:
                if row is None or 'word' not in row:
                    continue
                    
                word = row['word'].strip() if 'word' in row else None
                speaker = row['speaker'].strip() if 'speaker' in row else None
                
                if not word or not speaker:
                    continue
                
                # sil szavakat kihagyunk
                if word != 'sil':
                    words_list.append(word)
                    speakers.add(speaker)
                    total_frames += 1
                    
                    # Számlálás
                    if word in word_counts:
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
        
        # Unique szavak
        unique_words = sorted(list(set(words_list)))
        
        # Adatok formázása
        vocab_data = {
            "vocabulary": unique_words,
            "vocab_size": len(unique_words),
            "word_counts": word_counts,
            "total_frames": total_frames,
            "total_speakers": len(speakers),
            "speakers": sorted(list(speakers))
        }
        
        # JSON export
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✅ Szóvocabulárium generálva: {output_path}")
        print(f"{'='*70}")
        print(f"   📊 Unique szavak: {vocab_data['vocab_size']}")
        print(f"   📈 Total frame-ek (sil nélkül): {vocab_data['total_frames']}")
        print(f"   👥 Speakerek: {vocab_data['total_speakers']} ({', '.join(vocab_data['speakers'])})")
        
        print(f"\n📋 Top 15 leggyakoribb szó:")
        for i, (word, count) in enumerate(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15], 1):
            print(f"   {i:2d}. {word:15} {count:4d} frame")
        
        print(f"\n📄 Összes {vocab_data['vocab_size']} szó:")
        for i, word in enumerate(unique_words, 1):
            if (i - 1) % 5 == 0:
                print()
            print(f"   {word:15}", end="")
        print("\n")
        
        print(f"✅ A vocabulary.json-t a neurális háló tanításnál fogjuk használni!")
        print(f"{'='*70}\n")
        
        return vocab_data
        
    except FileNotFoundError as e:
        print(f"\n❌ Hiba: {e}")
        print(f"   CSV nem található: {csv_path}")
        print(f"   Futtasd előbb: python dataset_processor.py")
        print(f"   vagy: python dataset_processor_multithread.py\n")
        return None

if __name__ == "__main__":
    vocab = generate_vocabulary()
    sys.exit(0 if vocab else 1)
