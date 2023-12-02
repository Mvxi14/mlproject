# This specific file reads the dataset(s)
## Imports
# Because we need to use custom exception
import os 
import sys

# Things that we already created
from src.exception import CustomException
from src.logger import logging
import pandas as pd

# For splitting the dataset
from sklearn.model_selection import train_test_split

# Allows creating class variables
from dataclasses import dataclass

# Importing the transformation part we made
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# Importing the model training part we made
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# Decorator used here as its only definitions 
@dataclass
class DataIngestionConfig:
    # Here is where the data ingestion will save the files
    train_data_path:str = os.path.join('artifacts', "train.csv")
    test_data_path:str = os.path.join('artifacts', "test.csv")
    raw_data_path:str = os.path.join('artifacts', "data.csv")

# In this case we are not using the decorator, and we are using the definition by ourselves
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component") # Write in the log
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Dataset read as a dataframe")

            # Creating the path for train csv
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # Saving the original dataset 
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the train set
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Saving the test set
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed: Splitted and Saved")

            return(
                # Returning the path's for the data transformation process
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

