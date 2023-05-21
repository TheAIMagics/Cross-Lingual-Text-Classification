import sys
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
    
    def run_pipeline(self):
        try:
            self.data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e,sys)