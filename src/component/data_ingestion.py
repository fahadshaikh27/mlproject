import os
import sys
from src.exception import CustomExecption
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.component.data_transformation import DataTransFormation
from src.component.data_transformation import DataTransFormationConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initial_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # editing lines we can read from database or API
            df = pd.read_csv(r'D:\projects\mlproject\Notebook\data\stud.csv')

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path,
            )

        except Exception as e:
            raise CustomExecption(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, raw_data = obj.initial_data_ingestion()

    data_transformation = DataTransFormation()
    data_transformation.initiate_data_transformation(train_data, test_data)
