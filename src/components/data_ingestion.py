import os, sys
import numpy as np
from typing import List, Dict
#pytorch specific
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
# tensorflow specific
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
#imports from datasets
from datasets import  load_dataset, load_metric
#custom imports
from src.exception import CustomException
from src.logger import logging

model_URL = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
embed = hub.load(model_URL)

def embed_text(text : List[str]) -> List[np.ndarray]:
        """
        Embeds a list of text using a TensorFlow embedding model.

        Args:
            text (List[str]): List of text to be embedded.

        Returns:
            List[np.ndarray]: List of embedded vectors for each text.
        """
        with tf.device('/CPU:0'):
            vectors = embed(text)
            return [vector.numpy() for vector in vectors]

def encoder_factory(label2int: Dict[str,int]):
    """
    Factory function to create an encoder function for labeling and embedding data.

    Args:
        label2int (Dict[str, int]): Mapping of labels to integers.

    Returns:
        encode (function): Encoder function that takes a batch of data and performs encoding.
    """
    def encode(batch):
        batch['embedding'] = embed_text(batch['text'])
        batch['label'] = [label2int[str(x)] for x in batch['label']]
        return batch
    return encode

class SentimentDataloader(pl.LightningDataModule):
    """
    PyTorch Lightning data module for loading and preparing the Yelp Polarity dataset.

    Args:
        batch_size (int): Batch size for the dataloaders (default: 1).
        num_workers (int): Number of workers for data loading (default: 0).

    Attributes:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to use pinned memory for GPU tensors.
        test_ds: Dataset for the test set.
        train_ds: Dataset for the train set.
        val_ds: Dataset for the validation set.
        label_names: List of unique label names.
        encoder: Encoder function for labeling and embedding data.
        train: Train dataset mapped and batched.
        val: Validation dataset mapped and batched.
        test: Test dataset mapped and batched.

    Methods:
        prepare_data(): Prepare the data for training, validation, and testing.
        setup(): Setup the data for training, validation, and testing.
        train_dataloader(): Training dataloader.
        val_dataloader(): Validation dataloader.
        test_dataloader(): Test dataloader.
    """
    def __init__(
    self,
    batch_size :int =1,
    num_workers : int =0
):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()

    def prepare_data(self):
        """
        Prepare the data for training, validation, and testing.

        Loads the Yelp Polarity dataset and splits it into train, validation, and test sets.
        Encodes the labels using an encoder.

        Returns:
            None
        """
        self.test_ds = load_dataset('yelp_polarity', split="test[:2%]")
        self.train_ds = load_dataset('yelp_polarity', split="train[:2%]")
        self.val_ds = load_dataset('yelp_polarity', split="train[99%:]")

        self.label_names = self.train_ds.unique('label')
        label2int = {str(label):n for n,label in enumerate(self.label_names)}
        self.encoder = encoder_factory(label2int)

    def setup(self):
        """
        Setup the data for training, validation, and testing.

        Maps the encoded labels and batches the data using the specified batch size.
        Sets the format of the datasets to 'torch' and selects the 'embedding' and 'label' columns.

        Returns:
            None
        """
        self.train = self.train_ds.map(self.encoder,batched = True, batch_size = self.batch_size)
        self.train.set_format(type='torch', columns=['embedding','label'], output_all_columns= True)

        self.val = self.val_ds.map(self.encoder,batched = True, batch_size = self.batch_size)
        self.val.set_format(type='torch', columns=['embedding','label'], output_all_columns= True)

        self.test = self.test_ds.map(self.encoder,batched = True, batch_size = self.batch_size)
        self.test.set_format(type='torch', columns=['embedding','label'], output_all_columns= True)

    def train_dataloader(self):
        """
        This function returns a dataloader for the training set with specified batch size, number of
        workers, pin memory and shuffle.
        :return: A `torch.utils.data.DataLoader` object for the training set with the specified batch
        size, number of workers, pin memory, and shuffle settings.
        """
        """
        Training dataloader.

        Returns:
            torch.utils.data.DataLoader: Dataloader for the training set.
        """
        return DataLoader(self.train,                           
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        shuffle=True)

    def val_dataloader(self):
        """
        This function returns a DataLoader object for the validation dataset with specified batch size
        and number of workers.
        :return: The function `val_dataloader` is returning a `DataLoader` object that is created using
        the `val` dataset, with the specified batch size and number of workers.
        """
        
        return DataLoader(self.val,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        )

    def test_dataloader(self):
        """
        This function returns a DataLoader object for the test dataset with specified batch size and
        number of workers.
        :return: A `DataLoader` object is being returned with the test dataset, batch size, and number of
        workers specified in the arguments.
        """
        return DataLoader(self.test,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
    