#custom imports
import os,sys
from src.exception import CustomException
from src.logger import logging
from src.components.custom_model import YelpModel
from src.components.data_ingestion import SentimentDataloader
import pytorch_lightning as pl

S3_BUCKET_DATA_URI = 's3://multilingual-text/'

class ModelTrainer:

    def initiate_model_training(self):
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

