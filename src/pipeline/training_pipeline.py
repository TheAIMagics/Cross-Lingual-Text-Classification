import sys
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.model_trainer = ModelTrainer()
    
    def run_pipeline(self):
        try:
            # self.data_ingestion.initiate_data_ingestion()
            self.model_trainer.initiate_model_training()

        except Exception as e:
            raise CustomException(e,sys)