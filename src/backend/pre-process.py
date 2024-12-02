import os
import pandas as pd
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.preprocessing import clean_text


BASE_DIR = "data/raw/data/"
PROCESSED_DIR = "data/processed/"

def preprocess_files(directory, save_name):
    dir_path = os.path.join(BASE_DIR, directory)
    files = [f for f in os.listdir(dir_path) if f.endswith(".tsv") or f.endswith(".csv")]
    combined_data = []

    for file in files:
        file_path = os.path.join(dir_path, file)
        print(f"Processing {file}...")
        try:
            if file.endswith(".tsv"):
                df = pd.read_csv(file_path, sep="\t")
            else:
                df = pd.read_csv(file_path)
            df['cleaned_text'] = df['text'].apply(clean_text)
            combined_data.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    final_df = pd.concat(combined_data, ignore_index=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    final_df.to_csv(os.path.join(PROCESSED_DIR, save_name), index=False)
    print(f"Saved processed data to {os.path.join(PROCESSED_DIR, save_name)}")

if __name__ == "__main__":
    preprocess_files("all_data_en", "all_data_cleaned.csv")
