# All the different transformation needed in order to the ML model works correctly
import sys
import os
from dataclasses import dataclass

## Libraries for transformations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # Used to create the pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

## Exceptions and loggers written before
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl") # Any models used, saved into a pickle file in this path

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
        
        '''

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical values
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")), # Handling missing values
                    ("scaler", StandardScaler()) # Standardize the numerical features
                ]
            )

            logging.info("Numerical columns transformation completed: Imputations and Scaling")

            # Categorical values
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]

            )
            
            logging.info("Categorical columns transformation completed: Imputations, One-Hot Encoding and Scaling")

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Putting the num and cat columns transformation in sequence
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train an test data completed")

            logging.info("Obtaining preprocessing object")

            # Getting the transformer function we created inside this class
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writting_score", "reading_score"]

            # Splitting the target variable from the predictors
            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing function on training dataframe and testing dataframe")
            
            # Creating the train and test arrays with the preprocessing function
            input_fetaure_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_fetaure_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combining predictors with their respective target
            train_arr = np.c_[input_fetaure_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_fetaure_test_arr, np.array(target_feature_test_df)]


            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )
            logging.info(f"Saved preprocessing object.")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)

 