import os
import pandas as pd

BASE_DIR = "data/raw/data/"

def explore_directory(directory):
    """Explore files in a given directory."""
    print(f"Exploring {directory}...")
    dir_path = os.path.join(BASE_DIR, directory)
    files = [f for f in os.listdir(dir_path) if f.endswith(".tsv") or f.endswith(".csv")]

    for file in files:
        file_path = os.path.join(dir_path, file)
        print(f"Reading {file}...")
        try:
            if file.endswith(".tsv"):
                df = pd.read_csv(file_path, sep="\t")
            else:
                df = pd.read_csv(file_path)
            print(df.head())
        except Exception as e:
            print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    directories = [
        "all_data_en", "class_label_mapped", "data_split_all_lang",
        "event_aware_en", "individual_data_en", "initial_filtering"
    ]
    for directory in directories:
        explore_directory(directory)
