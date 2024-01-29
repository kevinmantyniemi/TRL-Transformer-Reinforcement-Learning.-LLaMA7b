from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    """
    A class for preprocessing datasets for the LLaMA model.

    Attributes:
        dataset_name (str): The name of the dataset to fetch and preprocess.
        tokenizer_model_name (str): The name of the tokenizer model to use for preprocessing.
        base_save_path (str): The base directory where the preprocessed data will be saved.
        subset_size (int, optional): The number of samples to consider from the dataset. If None, use the full dataset.

    Methods:
        preprocess_data(data): Tokenizes and preprocesses the given dataset split.
        fetch_and_preprocess(): Fetches the dataset, preprocesses it, and saves the output locally.
    """

    def __init__(self, dataset_name, tokenizer_model_name, base_save_path='data/datasets', subset_size=None):
        """
        Initializes the DataPreprocessor with the specified dataset and tokenizer.
        """
        self.dataset_name = dataset_name
        self.tokenizer_model_name = tokenizer_model_name  # Adding tokenizer_model_name as an attribute
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.base_save_path = base_save_path
        self.subset_size = subset_size
        
    def preprocess_data(self, data):
        """
        Preprocesses a given data split by tokenizing and applying other transformations.

        Args:
            data (Dataset): A dataset split to preprocess.

        Returns:
            Dataset: The preprocessed dataset split.
        """

        if self.subset_size:
            data = data.select(range(self.subset_size))

        def tokenize_function(examples):
            combined_text = [q + " " + a for q, a in zip(examples['Question'], examples['Answer'])]
            return self.tokenizer(combined_text, truncation=True, padding='max_length', max_length=128)

        return data.map(tokenize_function, batched=True)
                
    def fetch_and_preprocess(self):
        logging.info(f"Starting to fetch and preprocess dataset: {self.dataset_name}")
        try:
            print(dataset_name)
            dataset = load_dataset(self.dataset_name)
            print(dataset)
        except Exception as e:
            logging.error(f"Error fetching dataset '{self.dataset_name}': {e}")
            return

        save_path = os.path.join(self.base_save_path, 'data_processed', self.dataset_name.replace('/', '_'), self.tokenizer_model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for split in tqdm(dataset, desc="Processing splits"):
            try:
                processed_data = self.preprocess_data(dataset[split])
                processed_data_list = processed_data.to_dict()
                with open(os.path.join(save_path, f'{split}.json'), 'w') as file:
                    json.dump(processed_data_list, file, indent=4)
                logging.info(f"Processed and saved {split} split")
            except Exception as e:
                logging.error(f"Error processing split '{split}': {e}")


    def tokenize_function_selfrag(examples):
        # Combine 'Question' and 'Answer' fields for tokenization
        combined_text = [q + " " + a for q, a in zip(examples['Question'], examples['Answer'])]
        return self.tokenizer(combined_text, truncation=True, padding='max_length', max_length=128)
    
if __name__ == "__main__":
    dataset_name = 'selfrag/selfrag_train_data'  # Replace with your dataset
    tokenizer_model_name = 'gpt2'  # Use GPT-2 tokenizer
    subset_size = 1000  # Adjust the subset size as needed

    preprocessor = DataPreprocessor(dataset_name, tokenizer_model_name, subset_size=subset_size)
    preprocessor.fetch_and_preprocess()
