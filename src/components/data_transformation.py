import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path:str = os.path.join('artifacts',"preprocessor.pkl")
    # test_transform_path:str = os.path.join('transformed',"test_transform.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        logging.info("Data transformation is initiated")
        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = ['gender',
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns: {categorical_features}')
            logging.info(f'Numerical columns: {numerical_features}')

            preprocessor = ColumnTransformer(
                [
                    ('Numerical Pipeline',num_pipeline,numerical_features),
                    ('Categorical Pipeline',cat_pipeline,categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data')

            logging.info('Data Transformation Initiated')

            TARGET_COLUMN = 'math_score'
            # numerical_featuers = ['writing_score','reading_score']

            input_train_features = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_train_feature = train_df[TARGET_COLUMN]

            input_test_features = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_test_feature = test_df[TARGET_COLUMN]

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Data is ready to apply transformation")
            # preprocessing_obj.
         
            train_features_transformed = preprocessing_obj.fit_transform(input_train_features)
            test_features_transformed = preprocessing_obj.transform(input_test_features)

            train_arr = np.c_[
                train_features_transformed,np.array(target_train_feature)
            ]
            test_arr = np.c_[
                test_features_transformed,np.array(target_test_feature)
            ]
            save_object(
                file_path = self.transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )
           

            return (train_arr,test_arr,self.transformation_config.preprocessor_ob_file_path)
            


        except Exception as e:
            raise CustomException(e,sys)
            