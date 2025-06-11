import os
import csv
import pandas as pd
from tqdm import tqdm


DATASET_BASE_PATH = '/scratch/s5880432/whisper-finetune-mce/MCE_Dataset'


TEXT_DIR = os.path.join(DATASET_BASE_PATH, 'Text')
AUDIO_DIR = os.path.join(DATASET_BASE_PATH, 'Audio')
OUTPUT_CSV_PATH = './metadata.csv'

def create_metadata():
    
    all_data_rows = []
    print("--- Starting Metadata Creation (Based on Final Confirmed Logic) ---")
    print(f"Data Source Path: {DATASET_BASE_PATH}")
    print("-" * 60)

    
    for i in tqdm(range(1, 161), desc="Processing Speakers"):
        csv_path = os.path.join(TEXT_DIR, f'data_{i}.csv')
        speaker_audio_folder = os.path.join(AUDIO_DIR, f'{i}_MCE')

        if not os.path.exists(csv_path):
            continue
        if not os.path.exists(speaker_audio_folder):
            print(f"\nWarning: Audio folder not found for speaker {i}. Skipping.")
            continue

        try:
            with open(csv_path, mode='r', encoding='gb18030') as f:
                reader = csv.reader(f)
                
                
                try:
                    next(reader) 
                except StopIteration:
                    
                    continue
                
                for row_index, row in enumerate(reader, 1):
                    if not row or len(row) < 2:
                        continue 

                    
                    transcription = row[1].strip()
                    
                    
                    wav_filename = f"{i}_{row_index}.wav"
                    
                    
                    audio_path = os.path.join(speaker_audio_folder, wav_filename)

                    if os.path.exists(audio_path):
                        all_data_rows.append({'audio': audio_path, 'sentence': transcription})
                    else:
                        
                        print(f"\nWarning: Audio file not found at expected path: {audio_path}")

        except Exception as e:
            print(f"\nAn unexpected error occurred while processing {csv_path}: {e}")

    if not all_data_rows:
        print("\n--- ERROR: No metadata was generated. ---")
        return

    
    output_df = pd.DataFrame(all_data_rows)

    
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n--- SUCCESS! ---")
    print(f"Successfully created metadata file at: {OUTPUT_CSV_PATH}")
    print(f"Total entries found: {len(output_df)}")
    print("This is a sample of your final metadata:")
    print(output_df.head())

if __name__ == '__main__':
    create_metadata()