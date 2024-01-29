from datasets import load_dataset
import os

class DatasetHandler:
    def __init__(self, dataset_name, base_save_path='data/datasets', subset_size=None):
        self.dataset_name = dataset_name
        self.base_save_path = base_save_path
        self.subset_size = subset_size
        self.dataset = self.fetch_huggingface_dataset()

    def fetch_huggingface_dataset(self):
        """Fetch a dataset from Hugging Face and save a subset of it locally."""
        try:
            dataset = load_dataset(self.dataset_name)
        except Exception as e:
            print(f"Error fetching dataset '{self.dataset_name}': {e}")
            return None

        save_path = os.path.join(self.base_save_path, self.dataset_name.replace('/', '_'))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for split in dataset:
            try:
                subset = dataset[split].select(range(self.subset_size)) if self.subset_size else dataset[split]
                subset.to_csv(os.path.join(save_path, f'{split}.csv'))
            except Exception as e:
                print(f"Error processing split '{split}': {e}")

        return dataset
    
    
    

def main():
    dataset_name = 'selfrag/selfrag_train_data'  # Replace with your desired dataset name
    subset_size = 1000  # Number of samples per split; set to None for full dataset

    dataset_handler = DatasetHandler(dataset_name, subset_size=subset_size)
    if dataset_handler.dataset:
        print(f"Subset of dataset '{dataset_name}' saved and stored in object.")

if __name__ == "__main__":
    main()
