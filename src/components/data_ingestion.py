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
        with tf.device('/CPU:0'):
            vectors = embed(text)
            return [vector.numpy() for vector in vectors]

def encoder_factory(label2int: Dict[str,int]):
    def encode(batch):
        batch['embedding'] = embed_text(batch['text'])
        batch['label'] = [label2int[str(x)] for x in batch['label']]
        return batch
    return encode

class SentimentDataloader(pl.LightningDataModule):
    def __init__(
    self,
    batch_size :int =1,
    num_workers : int =0
):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()

    def prepare_data(self):
        self.test_ds = load_dataset('yelp_polarity', split="test[:2%]")
        self.train_ds = load_dataset('yelp_polarity', split="train[:2%]")
        self.val_ds = load_dataset('yelp_polarity', split="train[99%:]")

        self.label_names = self.train_ds.unique('label')
        label2int = {str(label):n for n,label in enumerate(self.label_names)}
        self.encoder = encoder_factory(label2int)

    def setup(self):
        self.train = self.train_ds.map(self.encoder,batched = True, batch_size = self.batch_size)
        self.train.set_format(type='torch', columns=['embedding','label'], output_all_columns= True)

        self.val = self.val_ds.map(self.encoder,batched = True, batch_size = self.batch_size)
        self.val.set_format(type='torch', columns=['embedding','label'], output_all_columns= True)

        self.test = self.test_ds.map(self.encoder,batched = True, batch_size = self.batch_size)
        self.test.set_format(type='torch', columns=['embedding','label'], output_all_columns= True)

    def train_dataloader(self):
        
        return DataLoader(self.train,                           
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        shuffle=True)

    def val_dataloader(self):
        
        return DataLoader(self.val,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        )

    def test_dataloader(self):
        return DataLoader(self.test,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
    