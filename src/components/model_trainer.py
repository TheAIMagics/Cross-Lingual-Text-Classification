#custom imports
import os,sys
from src.exception import CustomException
from src.logger import logging
from src.components.custom_model import YelpModel
from src.components.data_ingestion import SentimentDataloader
from src.cloud_storage.s3_operations import S3Sync
import pytorch_lightning as pl

S3_BUCKET_DATA_URI = 's3://multilingual-text/'

# The ModelTrainer class trains a sentiment analysis model using PyTorch Lightning and saves the best
# checkpoint to an S3 bucket.
class ModelTrainer:
    """
    Class for training and saving the model.

    Attributes:
        s3_sync: S3Sync object for syncing files to S3.

    Methods:
        __init__(): Initialize the ModelTrainer object.
        initiate_model_training(): Initiate the model training process.
    """
    def __init__(self):
        """
        This function initializes an object and tries to create an instance of the S3Sync class, raising
        a CustomException if there is an error.
        """
        try:
            self.s3_sync = S3Sync()
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self):
        """
        This function initiates the training of a sentiment analysis model using PyTorch Lightning and
        saves the best checkpoint to an S3 bucket.
        """
        try:
            data_loader = SentimentDataloader()
            data_loader.prepare_data()
            data_loader.setup()

            MAX_EPOCHS = 1
            model = YelpModel()
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath="model",
                filename="best_checkpoint",
                save_top_k=3,
                mode="min")

            trainer = pl.Trainer(
                accelerator="gpu",
                max_epochs=MAX_EPOCHS, 
                callbacks=[checkpoint_callback])
            
            trainer.fit(model, data_loader.train_dataloader(), data_loader.val_dataloader())

            self.s3_sync.sync_folder_to_s3(
                folder=model, aws_bucket_url=S3_BUCKET_DATA_URI)
            
        except Exception as e:
            raise CustomException(e, sys)

