import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join("artifacts","train.csv")
    test_data_path: str= os.path.join("artifacts","test.csv")
    raw_data_path: str= os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info("entered data ingestion method ")
        try:
            df = pd.read_csv(r"C:\Users\DELL\ML_Project\notebook\data\insurance.csv")
            logging.info("read dataset as dataframe ")

            os.makedirs(os.path.dirname(self.data_ingestion.train_data_path), exist_ok = True)
            df.to_csv(self.data_ingestion.raw_data_path,index = False , header = True)
            logging.info("raw data saved as csv file")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion.train_data_path,index= False,header = True)
            test_set.to_csv(self.data_ingestion.test_data_path,index=False,header = True)
            logging.info("test split completed")

            return (
                self.data_ingestion.train_data_path,
                self.data_ingestion.test_data_path
            ) 
        except Exception as e:
            raise CustomException(e,sys)

        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))        




