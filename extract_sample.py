#!/usr/bin/env python3
"""
mouth_data.csv első 1000 sorát kimenteni sample-ként
DEBUG és validáláshoz
"""

import csv
import sys

def extract_sample(input_csv="mouth_data.csv",
                   output_csv="mouth_data_sample_1000.csv",
                   sample_rows=1000):
    """
    CSV első N sorát kimenteni
    """
    
    print(f"\n📊 Sample CSV exportálás")
    print(f"   Input: {input_csv}")
    print(f"   Output: {output_csv}")
    print(f"   Sorok: {sample_rows}")
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as infile, \
             open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Header
            header = next(reader, None)
            if header is None:
                print("❌ Hiba: CSV nincs header!")
                return False
            
            writer.writerow(header)
            print(f"\n✅ Header kimentve ({len(header)} oszlop)")
            print(f"   Oszlopok: {', '.join(header[:5])}...")
            
            # Sorok
            row_count = 0
            for row in reader:
                if row_count >= sample_rows:
                    break
                writer.writerow(row)
                row_count += 1
                
                if (row_count + 1) % 100 == 0:
                    print(f"   ✓ {row_count} sor feldolgozva...")
            
            print(f"\n✅ Kész! {row_count} sor kimentve")
            print(f"   Output: {output_csv}")
            print(f"   Méret: {row_count} × {len(header)} (row × column)")
            
            return True
        
    except Exception as e:
        print(f"\n❌ Hiba: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔍 MOUTH_DATA SAMPLE EXPORT")
    print("="*70)
    
    success = extract_sample()
    
    if success:
        print("\n" + "="*70)
        print("✅ Sample CSV sikeresen létrehozva!")
        print("   Most megnyithatod Excelben: mouth_data_sample_1000.csv")
        print("="*70 + "\n")
    else:
        print("\n❌ Hiba az exportálás során!\n")
        sys.exit(1)
