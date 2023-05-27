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
#imports from datasets
from datasets import load_metric
#custom imports
from src.exception import CustomException
from src.logger import logging

class YelpModel(pl.LightningModule):
  def __init__(self,
               hidden_dims : List[int] = [768,128],
               dropout_prob : float = 0.5,
               learning_rate : float = 1e-3
               ):
    super().__init__()
    self.train_acc = load_metric("accuracy")
    self.val_acc = load_metric("accuracy")
    self.test_acc = load_metric("accuracy")
    self.hidden_dims = hidden_dims
    self.dropout_prob = dropout_prob
    self.learning_rate = learning_rate

    self.embedding_dim = 512

    layers = []
    prev_dim = self.embedding_dim

    if dropout_prob > 0:
        layers.append(nn.Dropout(dropout_prob))

    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        prev_dim = h
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.ReLU())
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
    # output layer
    layers.append(nn.Linear(prev_dim, 2))

    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    # x will be a batch of USEm vectors
    logits = self.layers(x)
    return logits

  def configure_optimizers(self):
  
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # type: ignore
    return optimizer

  def __compute_loss(self, batch):
    
    x, y = batch["embedding"], batch["label"]
    logits = self(x)
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    loss = F.cross_entropy(logits, y)
    return loss, preds, y

  def training_step(self, batch, batch_idx):
    loss, preds, y = self.__compute_loss(batch)
    self.train_acc.add_batch(predictions=preds, references=y)
    acc = self.train_acc.compute()["accuracy"] # type: ignore
    values = {"train_loss": loss, "train_accuracy": acc}
    self.log_dict(values, on_step=True, on_epoch=True,
                  prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    loss, preds, y = self.__compute_loss(batch)
    self.val_acc.add_batch(predictions=preds, references=y)
    acc = self.val_acc.compute()["accuracy"]    # type: ignore
    values = {"val_loss": loss, "val_accuracy": acc}
    self.log_dict(values, on_step=True, on_epoch=True,
                  prog_bar=True, logger=True)
    return loss


  def test_step(self, batch, batch_idx):
    loss, preds, y = self.__compute_loss(batch)
    self.test_acc.add_batch(predictions=preds, references=y)
    acc = self.test_acc.compute()["accuracy"]   # type: ignore
    values = {"test_loss": loss, "test_accuracy": acc}
    self.log_dict(values, on_step=False, on_epoch=True,
                  prog_bar=True, logger=True)
    return loss