import os
import sys
import pickle
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:  #mapping html backend input with values
    def __init__(self,
        Gender: str,
        Age: int,
        Driving_License: int,
        Region_Code: int,
        Previously_Insured: int,
        Vehicle_Age: str,
        Vehicle_Damage: str,
        Annual_Premium: float,
        Policy_Sales_Channel: int,
        Vintage: int):
        self.Gender = Gender   # input user value
        self.Age = Age
        self.Driving_License = Driving_License
        self.Region_Code = Region_Code  
        self.Previously_Insured = Previously_Insured
        self.Vehicle_Age = Vehicle_Age
        self.Vehicle_Damage = Vehicle_Damage
        self.Annual_Premium = Annual_Premium
        self.Policy_Sales_Channel = Policy_Sales_Channel
        self.Vintage = Vintage
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

                 
    


