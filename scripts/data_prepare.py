import zipfile
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.sparse import csr_matrix
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def extract_ziped_data(ziped_data_path: str, extract_path : str):
    """Extracts the contents of a zip file to a specified directory.
    
    args:
        ziped_data_path: str, path to the zip file
        extract_path: str, path to the directory where contents will be extracted
    """
    # The directory where you want to extract the contents
    extract_path = 'data'

    # Open the zip file in read mode
    with zipfile.ZipFile(ziped_data_path, 'r') as zip_ref:
        # Extract all the contents into the specified directory
        zip_ref.extractall(extract_path)

    print(f"'{ziped_data_path}' has been extracted to '{extract_path}'")

def prepare_data(data_folder='data/', val_days=7, test_days=7):
    """
    Loads, preprocesses, and splits the events data into train, validation, and test sets.
    
    args:
        data_folder: str, path to the folder containing 'events.csv'
        val_days: int, number of days for the validation set
        test_days: int, number of days for the test set
    """
    # --- Load Data ---
    print(f"Loading events.csv from folder: {data_folder}")
    try:
        events_df = pd.read_csv(data_folder + 'events.csv')
        print("Successfully loaded events.csv.")
        events_df['timestamp_dt'] = pd.to_datetime(events_df['timestamp'], unit='ms')
        print("\n--- Initial Data Summary ---")
        print(f"Data shape: {events_df.shape}")
        print(f"Full timeframe: {events_df['timestamp_dt'].min()} to {events_df['timestamp_dt'].max()}")
        print("----------------------------\n")
    except FileNotFoundError:
        print(f"Error: 'events.csv' not found in '{data_folder}'. Please check the path.")
        return None, None, None

    # --- Split Data ---
    sorted_df = events_df.sort_values('timestamp_dt').reset_index(drop=True)
    print(f"Splitting data: {test_days} days for test, {val_days} for validation.")
    end_time = sorted_df['timestamp_dt'].max()
    test_start_time = end_time - timedelta(days=test_days)
    val_start_time = test_start_time - timedelta(days=val_days)

    test_df = sorted_df[sorted_df['timestamp_dt'] >= test_start_time]
    val_df = sorted_df[(sorted_df['timestamp_dt'] >= val_start_time) & (sorted_df['timestamp_dt'] < test_start_time)]
    train_df = sorted_df[sorted_df['timestamp_dt'] < val_start_time]

    print("--- Data Splitting Summary ---")
    print(f"Training set:   {train_df.shape[0]:>8} records | from {train_df['timestamp_dt'].min()} to {train_df['timestamp_dt'].max()}")
    print(f"Validation set: {val_df.shape[0]:>8} records | from {val_df['timestamp_dt'].min()} to {val_df['timestamp_dt'].max()}")
    print(f"Test set:       {test_df.shape[0]:>8} records | from {test_df['timestamp_dt'].min()} to {test_df['timestamp_dt'].max()}")
    print("------------------------------")
    
    return train_df, val_df, test_df

class SASRecDataset(Dataset):
    """
    SASRec Dataset.
    - Precomputes (sequence_id, cutoff_idx) pairs for O(1) __getitem__.
    - Supports 'last' or 'all' target modes.
    """
    def __init__(self, sequences, max_len, target_mode="last"):
        """
        Args:
            sequences: list of user sequences (list of item IDs).
            max_len: maximum sequence length (padding applied).
            target_mode: 'last' (only last prediction) or 'all' (predict at every step).
        """
        self.sequences = sequences
        self.max_len = max_len
        self.target_mode = target_mode

        # Build index once
        self.index = []
        for seq_id, seq in enumerate(sequences):
            for i in range(1, len(seq)):
                self.index.append((seq_id, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_id, cutoff = self.index[idx]
        seq = self.sequences[seq_id][:cutoff]

        # Truncate & pad
        seq = seq[-self.max_len:]
        pad_len = self.max_len - len(seq)

        input_seq = np.zeros(self.max_len, dtype=np.int64)
        input_seq[pad_len:] = seq

        if self.target_mode == "last":
            target = self.sequences[seq_id][cutoff]
            return torch.LongTensor(input_seq), torch.LongTensor([target])

        elif self.target_mode == "all":
            # Predict next item at each step
            target_seq = self.sequences[seq_id][1:cutoff+1]
            target_seq = target_seq[-self.max_len:]
            target = np.zeros(self.max_len, dtype=np.int64)
            target[-len(target_seq):] = target_seq
            return torch.LongTensor(input_seq), torch.LongTensor(target)

class SASRecDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for preparing the RetailRocket dataset for the SASRec model.

    This class handles all aspects of data preparation, including:
    - Filtering out infrequent users and items to reduce noise.
    - Building a consistent item vocabulary.
    - Converting user event histories into sequential data.
    - Creating and providing `DataLoader` instances for training, validation, and testing.
    """
    def __init__(self, train_df, val_df, test_df, min_item_interactions=5, 
                 min_user_interactions=5, max_len=50, batch_size=256):
        """
        Initializes the DataModule.

        Args:
            train_df (pd.DataFrame): DataFrame for training.
            val_df (pd.DataFrame): DataFrame for validation.
            test_df (pd.DataFrame): DataFrame for testing.
            min_item_interactions (int): Minimum number of interactions for an item to be kept.
            min_user_interactions (int): Minimum number of interactions for a user to be kept.
            max_len (int): The maximum length of a user sequence fed to the model.
            batch_size (int): The batch size for the DataLoaders.
        """
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.min_item_interactions = min_item_interactions
        self.min_user_interactions = min_user_interactions
        self.max_len = max_len
        self.batch_size = batch_size

        self.item_map = None
        self.inverse_item_map = None
        self.vocab_size = 0
        self.user_history = None

    def setup(self, stage=None):
        """
        Prepares the data for training, validation, and testing.

        This method is called automatically by PyTorch Lightning. It performs the following steps:
        1. Determines filtering criteria (which users and items to keep) based on the training set only
           to prevent data leakage.
        2. Applies these filters to the train, validation, and test sets.
        3. Builds an item vocabulary (mapping item IDs to integer indices) from the combined
           training and validation sets to ensure consistency for model checkpointing.
        4. Converts the event logs into sequences of item indices for each user in each data split.
        """
        item_counts = self.train_df['itemid'].value_counts()
        user_counts = self.train_df['visitorid'].value_counts()
        items_to_keep = item_counts[item_counts >= self.min_item_interactions].index
        users_to_keep = user_counts[user_counts >= self.min_user_interactions].index

        self.filtered_train_df = self.train_df[
            (self.train_df['itemid'].isin(items_to_keep)) & 
            (self.train_df['visitorid'].isin(users_to_keep))
        ].copy()
        self.filtered_val_df = self.val_df[
            (self.val_df['itemid'].isin(items_to_keep)) & 
            (self.val_df['visitorid'].isin(users_to_keep))
        ].copy()
        self.filtered_test_df = self.test_df[
            (self.test_df['itemid'].isin(items_to_keep)) & 
            (self.test_df['visitorid'].isin(users_to_keep))
        ].copy()

        all_known_items_df = pd.concat([self.filtered_train_df, self.filtered_val_df])
        unique_items = all_known_items_df['itemid'].unique()
        self.item_map = {item_id: i + 1 for i, item_id in enumerate(unique_items)}
        self.inverse_item_map = {i: item_id for item_id, i in self.item_map.items()}
        self.vocab_size = len(self.item_map) + 1 # +1 for padding token 0

        self.user_history = self.filtered_train_df.groupby('visitorid')['itemid'].apply(list)
        
        self.train_sequences = self._create_sequences(self.filtered_train_df)
        self.val_sequences = self._create_sequences(self.filtered_val_df)
        self.test_sequences = self._create_sequences(self.filtered_test_df)

    def _create_sequences(self, df):
        """
        Helper function to convert a DataFrame of events into user interaction sequences.
        
        Args:
            df (pd.DataFrame): The input DataFrame to process.

        Returns:
            list[list[int]]: A list of user sequences, where each sequence is a list of item indices.
        """
        df_sorted = df.sort_values(['visitorid', 'timestamp_dt'])
        sequences = df_sorted.groupby('visitorid')['itemid'].apply(
            lambda x: [self.item_map[i] for i in x if i in self.item_map]
        ).tolist()
        return [s for s in sequences if len(s) > 1]

    def train_dataloader(self):
        """Creates the DataLoader for the training set."""
        dataset = SASRecDataset(self.train_sequences, self.max_len)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        """Creates the DataLoader for the validation set."""
        dataset = SASRecDataset(self.val_sequences, self.max_len)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self):
        """Creates the DataLoader for the test set."""
        dataset = SASRecDataset(self.test_sequences, self.max_len)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

if __name__ == "__main__":
    
    # --- Configuration ---
    DATA_PATH = "data" 
    ZIPED_DATA_PATH = "data/archive.zip" # change to your zip file path
    BATCH_SIZE = 256       
    MAX_TOKEN_LEN = 50     # 50–100 is standard for SASRec
    
    # extract_ziped_data(ZIPED_DATA_PATH, DATA_PATH) # uncomment this line if you want to extract the data
    
    # --- 1. Prepare the data into train, validation, and test sets ---
    train_set, validation_set, test_set = prepare_data(data_folder=DATA_PATH)

    # --- 2. Initialize DataModule ---
    print("Initializing DataModule...")
    datamodule = SASRecDataModule(
        train_df=train_set,
        val_df=validation_set,
        test_df=test_set,
        batch_size=BATCH_SIZE,
        max_len=MAX_TOKEN_LEN
    )
    datamodule.setup()