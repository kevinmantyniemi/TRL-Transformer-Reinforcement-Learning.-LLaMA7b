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

    def __init__(self, dataset_name, tokenizer_model_name, splits=None, base_save_path='data/datasets', subset_size=None):
        self.dataset_name = dataset_name
        self.tokenizer_model_name = tokenizer_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.base_save_path = base_save_path
        self.subset_size = subset_size
        self.splits = splits if splits is not None else ['train', 'test', 'validation']

        
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

        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return data.map(self.tokenize_function, batched=True)

    
    
    def tokenize_function(self, examples):
        tokenized_output = {}
        for field in self.text_fields:
            # Apply tokenizer and convert to standard Python lists
            tokenized_batch = self.tokenizer(examples[field], truncation=True, padding='max_length', max_length=128)
            tokenized_output[field] = tokenized_batch['input_ids']
        return tokenized_output

    def identify_text_fields(self, dataset):
        # Identifying text fields in the dataset
        sample = next(iter(dataset))
        self.text_fields = [key for key, value in sample.items() if isinstance(value, str)]

    def fetch_and_preprocess(self):
        logging.info(f"Starting to fetch and preprocess dataset: {self.dataset_name}")
        try:
            dataset = load_dataset(self.dataset_name)
            logging.info(f"Fetched dataset: {self.dataset_name}")
        except Exception as e:
            logging.error(f"Error fetching dataset '{self.dataset_name}': {e}")
            return

        # Create a directory for the raw dataset
        raw_data_path = os.path.join(self.base_save_path, self.dataset_name)
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
            logging.info(f"Created directory for raw dataset: {raw_data_path}")

        # Save the raw dataset
        for split in dataset:
            raw_split_path = os.path.join(raw_data_path, f'{split}.json')
            with open(raw_split_path, 'w') as file:
                json.dump(dataset[split].to_dict(), file, indent=4)
            logging.info(f"Saved raw {split} split to {raw_split_path}")

        # Process and save the dataset
        save_path = os.path.join(self.base_save_path, 'data_processed', self.dataset_name.replace('/', '_'), self.tokenizer_model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logging.info(f"Created directory: {save_path}")

        for split in tqdm(self.splits, desc="Processing splits"):
            if split in dataset:
                logging.info(f"Processing split: {split}")
                try:
                    self.identify_text_fields(dataset[split])
                    processed_data = self.preprocess_data(dataset[split])
                    processed_data_list = processed_data.to_dict()
                    file_path = os.path.join(save_path, f'{split}.json')
                    with open(file_path, 'w') as file:
                        json.dump(processed_data_list, file, indent=4)
                    logging.info(f"Processed and saved {split} split to {file_path}")
                except Exception as e:
                    logging.error(f"Error processing split '{split}': {e}")



if __name__ == "__main__":
    dataset_name = 'selfrag/selfrag_train_data'  # Replace with your dataset
    #dataset_name = 'imdb'
    tokenizer_model_name = 'zypher'  # Use GPT-2 tokenizer
    subset_size = 2  # Adjust the subset size as needed

    preprocessor = DataPreprocessor(dataset_name, tokenizer_model_name, splits=['train', 'test'], subset_size=subset_size)
    preprocessor.fetch_and_preprocess()
