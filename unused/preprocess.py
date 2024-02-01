import pandas as pd
import os

def load_data(file_path):
    """Load dataset from a file."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Example: Fill missing values with the mean of the column
    return df.fillna(df.mean())

def preprocess_data(file_path):
    """Preprocess the dataset."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    # Add additional preprocessing steps as needed
    return df

def main():
    dataset_name = 'selfrag_selfrag_train_data'  # Adjust as per your dataset
    split = 'train'  # Adjust as per the specific split you want to preprocess
    file_path = f'data/datasets/{dataset_name}/{split}.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    print(f"Preprocessing the dataset from {file_path}...")
    df = preprocess_data(file_path)
    print("Preprocessing completed.")

    # Optionally, save the preprocessed dataset
    preprocessed_file_path = f'data/datasets/{dataset_name}/{split}_preprocessed.csv'
    df.to_csv(preprocessed_file_path, index=False)
    print(f"Preprocessed dataset saved to {preprocessed_file_path}")

if __name__ == "__main__":
    main()
