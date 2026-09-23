import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def custom_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the custom preprocessing required for
        the Vehicle Insurance dataset.
        """

        try:

            logging.info("Starting custom preprocessing")

            df = df.copy()

            # --------------------------------------------------
            # 1. Convert numerical columns to numeric datatype
            # --------------------------------------------------

            numeric_columns = [
                "Age",
                "Driving_License",
                "Region_Code",
                "Previously_Insured",
                "Annual_Premium",
                "Policy_Sales_Channel",
                "Vintage"
            ]

            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            # --------------------------------------------------
            # 2. Drop ID column
            # --------------------------------------------------

            if "id" in df.columns:
                df = df.drop(columns=["id"])

            # --------------------------------------------------
            # 3. Convert Gender into binary values
            # Female -> 0
            # Male   -> 1
            # --------------------------------------------------

            df["Gender"] = df["Gender"].map({
                "Female": 0,
                "Male": 1
            }).astype(int)

            # --------------------------------------------------
            # 4. One-Hot Encoding
            # Vehicle_Age
            # Vehicle_Damage
            # --------------------------------------------------

            categorical_columns = [
                "Vehicle_Age",
                "Vehicle_Damage"
            ]

            df = pd.get_dummies(
                df,
                columns=categorical_columns,
                drop_first=True
            )

            # --------------------------------------------------
            # 5. Rename generated columns
            # --------------------------------------------------

            df.rename(
                columns={
                    "Vehicle_Age_< 1 Year":
                        "Vehicle_Age_lt_1_Year",

                    "Vehicle_Age_> 2 Years":
                        "Vehicle_Age_gt_2_Years"
                },
                inplace=True
            )

            # --------------------------------------------------
            # 6. Make sure all expected columns exist
            # --------------------------------------------------

            expected_columns = [
                "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_gt_2_Years",
                "Vehicle_Damage_Yes"
            ]

            for col in expected_columns:

                if col not in df.columns:
                    df[col] = 0

            # --------------------------------------------------
            # 7. Keep exact feature order
            # --------------------------------------------------

            df = df.reindex(
                columns=[
                    "Gender",
                    "Age",
                    "Driving_License",
                    "Region_Code",
                    "Previously_Insured",
                    "Annual_Premium",
                    "Policy_Sales_Channel",
                    "Vintage",
                    "Vehicle_Age_lt_1_Year",
                    "Vehicle_Age_gt_2_Years",
                    "Vehicle_Damage_Yes"
                ]
            )

            logging.info("Custom preprocessing completed")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        Creates the preprocessing object for the
        Vehicle Insurance dataset.
        """

        try:

            # --------------------------------------------------
            # Columns for StandardScaler
            # --------------------------------------------------

            numerical_columns = [
                "Age",
                "Vintage"
            ]

            # --------------------------------------------------
            # Columns for MinMaxScaler
            # --------------------------------------------------

            minmax_columns = [
                "Annual_Premium"
            ]

            logging.info(
                f"StandardScaler columns: {numerical_columns}"
            )

            logging.info(
                f"MinMaxScaler columns: {minmax_columns}"
            )

            # --------------------------------------------------
            # Preprocessing pipeline
            #
            # IMPORTANT:
            # custom_preprocessing happens FIRST.
            # Scaling happens AFTER custom preprocessing.
            # --------------------------------------------------

            preprocessing_pipeline = Pipeline(
                steps=[

                    (
                        "custom_preprocessing",
                        FunctionTransformer(
                            self.custom_preprocessing,
                            validate=False
                        )
                    ),

                    (
                        "scaling",
                        ColumnTransformer(
                            transformers=[

                                (
                                    "standard_scaler",
                                    StandardScaler(),
                                    numerical_columns
                                ),

                                (
                                    "minmax_scaler",
                                    MinMaxScaler(),
                                    minmax_columns
                                )

                            ],
                            remainder="passthrough"
                        )
                    )

                ]
            )

            logging.info(
                "Vehicle Insurance preprocessing pipeline created"
            )

            return preprocessing_pipeline

        except Exception as e:

            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            # --------------------------------------------------
            # 1. Read train and test data
            # --------------------------------------------------

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Read train and test data completed"
            )

            # --------------------------------------------------
            # 2. Obtain preprocessing object
            # --------------------------------------------------

            logging.info(
                "Obtaining preprocessing object"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            # --------------------------------------------------
            # 3. Target column
            # --------------------------------------------------

            target_column_name = "Response"

            # --------------------------------------------------
            # 4. Separate input features and target
            # --------------------------------------------------

            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            # --------------------------------------------------
            # 5. Apply preprocessing
            # --------------------------------------------------

            logging.info(
                "Applying preprocessing object on "
                "training and testing data"
            )

            # custom_preprocessing()
            #       ↓
            # StandardScaler / MinMaxScaler
            #       ↓
            # transformed train data

            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            # Test data only gets transform()
            # It does NOT fit the preprocessing object again.

            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            logging.info(
                "Preprocessing completed successfully"
            )

            # --------------------------------------------------
            # 6. Handle class imbalance using SMOTEENN
            # --------------------------------------------------

            logging.info(
                "Applying SMOTEENN to training data"
            )

            smt = SMOTEENN(
                sampling_strategy="minority"
            )

            input_feature_train_final, \
            target_feature_train_final = smt.fit_resample(
                input_feature_train_arr,
                target_feature_train_df
            )

            # --------------------------------------------------
            # 7. SMOTEENN on test data
            # --------------------------------------------------
            # Kept here to preserve your original code's
            # existing behavior.
            # --------------------------------------------------

            input_feature_test_final, \
            target_feature_test_final = smt.fit_resample(
                input_feature_test_arr,
                target_feature_test_df
            )

            logging.info(
                "SMOTEENN completed"
            )

            # --------------------------------------------------
            # 8. Combine features and target
            # --------------------------------------------------

            train_arr = np.c_[
                input_feature_train_final,
                np.array(target_feature_train_final)
            ]

            test_arr = np.c_[
                input_feature_test_final,
                np.array(target_feature_test_final)
            ]

            # --------------------------------------------------
            # 9. Save preprocessing object
            # --------------------------------------------------

            logging.info(
                "Saving preprocessing object"
            )

            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path),
                obj=preprocessing_obj
            )

            logging.info(
                "Preprocessing object saved successfully"
            )

            # --------------------------------------------------
            # 10. Return results
            # --------------------------------------------------

            return (
                train_arr,
                test_arr,
                self.data_transformation_config
                .preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)