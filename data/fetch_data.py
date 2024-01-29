from datasets import load_dataset
import os

def fetch_huggingface_dataset(dataset_name, base_save_path='data/datasets', subset_size=None):
    """Fetch a dataset from Hugging Face and save a subset of it locally."""
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error fetching dataset '{dataset_name}': {e}")
        return

    save_path = os.path.join(base_save_path, dataset_name.replace('/', '_'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for split in dataset:
        try:
            # Select a subset if subset_size is specified
            subset = dataset[split].select(range(subset_size)) if subset_size else dataset[split]
            subset.to_csv(os.path.join(save_path, f'{split}.csv'))
        except Exception as e:
            print(f"Error processing split '{split}': {e}")

def main():
    dataset_name = 'selfrag/selfrag_train_data'  # Replace with your desired dataset name
    subset_size = 1000                            # Number of samples per split; set to None for full dataset

    print(f"Fetching dataset '{dataset_name}' from Hugging Face...")
    fetch_huggingface_dataset(dataset_name, subset_size=subset_size)
    print(f"Subset of dataset saved to '{dataset_name}'")

if __name__ == "__main__":
    main()
