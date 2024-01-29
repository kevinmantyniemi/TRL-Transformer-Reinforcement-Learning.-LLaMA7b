from datasets import load_dataset
from transformers import AutoTokenizer
import os

class DataPreprocessor:
    def __init__(self, dataset_name, tokenizer_model_name, base_save_path='data/datasets', subset_size=None):
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.base_save_path = base_save_path
        self.subset_size = subset_size

    def preprocess_data(self, data):
        """Preprocess data for LLaMA model."""
        if self.subset_size:
            data = data.select(range(self.subset_size))

        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

        tokenized_data = data.map(tokenize_function, batched=True)
        return tokenized_data

    def fetch_and_preprocess(self):
        """Fetch a dataset from Hugging Face, preprocess, and save locally."""
        dataset = load_dataset(self.dataset_name)
        save_path = os.path.join(self.base_save_path, self.dataset_name.replace('/', '_'))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for split in dataset:
            processed_data = self.preprocess_data(dataset[split])
            # Save your processed data as needed

if __name__ == "__main__":
    dataset_name = 'your_dataset_name'  # Replace with your dataset
    tokenizer_model_name = 'your_model_name'  # Replace with the tokenizer model
    subset_size = 1000  # Adjust the subset size as needed

    preprocessor = DataPreprocessor(dataset_name, tokenizer_model_name, subset_size=subset_size)
    preprocessor.fetch_and_preprocess()
