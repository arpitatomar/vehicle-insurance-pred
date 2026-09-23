import os
import sys
import pickle

from src.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using pickle.

    :param file_path: The path where the object will be saved.
    :param obj: The Python object to be saved.
    :raises CustomException: If an error occurs during the saving process.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


