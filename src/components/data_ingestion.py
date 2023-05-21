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
from datasets import Dataset, load_dataset, load_metric
#custom imports
from src.exception import CustomException
from src.components.custom_data import SentimentDataloader
from src.logger import logging

model_URL = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
embed = hub.load(model_URL)

class DataIngestion:

    def initiate_data_ingestion(self):
        try:
            data_loader = SentimentDataloader()
            data_loader.prepare_data()
            data_loader.setup()
            print(len(data_loader.train))
            print(len(data_loader.val))
            print(len(data_loader.test))
        except Exception as e:
            raise CustomException(e, sys)