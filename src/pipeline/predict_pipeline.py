import pandas as pd
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class CustomData:
    def __init__(self,
                 age: float,
                 sex: str,
                 bmi: float,
                 children: int,
                 smoker: str,
                 region: str):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        
    def predict(self, features):
        try:
            logging.info("Loading model and preprocessor")
            
            # Load the model and preprocessor
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            logging.info("Model and preprocessor loaded successfully")
            
            # Transform the data using the preprocessor
            data_scaled = preprocessor.transform(features)
            
            logging.info("Data preprocessing completed")
            
            # Make prediction
            preds = model.predict(data_scaled)
            
            logging.info(f"Prediction made: {preds[0]}")
            
            return preds
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)