import os,sys
import torch
import numpy as np
from typing import List, Dict
import tensorflow as tf
import tensorflow_hub as hub
from src.logger import logging
from src.exception import CustomException

from src.components.custom_model import YelpModel
from src.cloud_storage.s3_operations import S3Sync
model_URL = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
embed = hub.load(model_URL)


S3_BUCKET_DATA_URI = 's3://multilingual-text/'
class SinglePrediction:
    def __init__(self):
            try:
                self.s3_sync = S3Sync()
            except Exception as e:
                raise CustomException(e, sys)
            
    
    def embed_text(self,text : List[str]) -> List[np.ndarray]:
            with tf.device('/CPU:0'):
                vectors = embed(text)
                return [vector.numpy() for vector in vectors]

    def get_model_in_production(self):
        try:
            model_download_dir = os.path.join(os.getcwd(),'static')
            os.makedirs(model_download_dir,exist_ok=True)
            logging.info(f"Model Directory created at {model_download_dir}")
            self.s3_sync.sync_folder_from_s3(folder=model_download_dir, aws_bucket_url=S3_BUCKET_DATA_URI)
            #Model Directory is empty
            if not any(os.scandir(model_download_dir)):
                self.s3_sync.sync_folder_from_s3(folder=model_download_dir, aws_bucket_url=S3_BUCKET_DATA_URI)
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self,text: List[str]):
        try:
            self.get_model_in_production()
            model_download_dir = os.path.join(os.getcwd(),'static','model','model.pt')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            model = YelpModel()
            model_state_dict = torch.load(model_download_dir,map_location=device)
            model.load_state_dict(model_state_dict['model_state_dict'])
            

            embeddings = torch.Tensor(self.embed_text(text)).to(model.device)
            logits = model(embeddings)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            scores = torch.softmax(logits, dim=1).detach().cpu().numpy()

            results = []
            for t, best_index, score_pair in zip(text, preds, scores):
                results.append({
                    "text": t,
                    "label": "positive" if best_index == 1 else "negative",
                    "score": score_pair[best_index]
                })
            return results
            
        except Exception as e:
            raise CustomException(e, sys)