import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Initiated")
        try:
            df = pd.read_csv('notebook/dataset/student.csv')
            logging.info("Data Ingested Successfully")

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path),exist_ok=True)

            df.to_csv(self.ingestionConfig.raw_data_path,index=False,header=True)

            logging.info("Train and Test Data Splitting Initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionConfig.test_data_path,index=False,header=True)

            logging.info("Data Ingstion Completed")
            return (
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path,
                self.ingestionConfig.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data,raw_data = obj.initiate_data_ingestion()

    obj1 = DataTransformation()
    train_arr,test_arr,_ = obj1.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))