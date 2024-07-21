import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "/home/naveed-ur-rehman/Desktop/mlproject/artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education:str,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    # def get_data_as_df(self):
    #     try:
    #         custom_data_input_dict = {
    #             "gender":[self.gender],
    #             "race_ethnicity":[self.race_ethnicity],
    #             "parental_level_of_education":[self.parental_level_of_education],
    #             "lunch":[self.lunch],
    #             "test_preparation_course":[self.test_preparation_course],
    #             "reading_score":[self.reading_score],
    #             "writing_score":[self.writing_score]
    #         }

    #         return pd.DataFrame(custom_data_input_dict)
    #     except Exception as e:
    #         raise CustomException(e,sys)
            
    def get_data_as_df(self):
        try:
            # Define default values for each attribute
            default_values = {
                "gender": "male",
                "race_ethnicity": "group A",
                "parental_level_of_education": "associate's degree",
                "lunch": "standard",
                "test_preparation_course": "completed",
                "reading_score": 0,  # Assuming 0 is a sensible default
                "writing_score": 0   # Assuming 0 is a sensible default
            }

            # Prepare the data input dictionary, replacing None values with defaults
            custom_data_input_dict = {
                "gender": [self.gender or default_values["gender"]],
                "race_ethnicity": [self.race_ethnicity or default_values["race_ethnicity"]],
                "parental_level_of_education": [self.parental_level_of_education or default_values["parental_level_of_education"]],
                "lunch": [self.lunch or default_values["lunch"]],
                "test_preparation_course": [self.test_preparation_course or default_values["test_preparation_course"]],
                "reading_score": [self.reading_score if self.reading_score is not None else default_values["reading_score"]],
                "writing_score": [self.writing_score if self.writing_score is not None else default_values["writing_score"]]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)


