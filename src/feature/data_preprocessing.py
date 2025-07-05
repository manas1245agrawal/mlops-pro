from pathlib import Path
import sys
import typer
from yaml import safe_load
import joblib
import pandas as pd
import warnings
import sklearn
from loguru import logger
from utils import train_preprocessor, transform_data, save_dataframe, save_transformer, read_dataframe

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output="pandas")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

@app.command()
def main():
    logger.info("Pipeline started.")
    input_path: Path = PROCESSED_DATA_DIR

    logger.info("Reading parameters from params.yaml")
    with open('params.yaml') as f:
        params = safe_load(f)

    save_transformers_path = PROJ_ROOT / 'models' / 'transformers'
    save_transformers_path.mkdir(exist_ok=True)
    logger.info(f"Transformer save path created or already exists at {save_transformers_path}")

    save_data_path = PROCESSED_DATA_DIR / 'final'
    save_data_path.mkdir(exist_ok=True)
    logger.info(f"Data save path created or already exists at {save_data_path}")

    for filename in ['train.csv', 'val.csv','test.csv']:
        complete_input_path = input_path / filename
        logger.info(f"Processing file: {filename}")

        if filename == 'train.csv':
            df = read_dataframe(complete_input_path)
            logger.info(f"Train dataframe loaded with shape {df.shape}")

            preprocessor = train_preprocessor(data=df)
            logger.info("Preprocessor trained on train data")

            save_transformer(
                path=save_transformers_path / 'preprocessor.joblib',
                object=preprocessor
            )
            logger.info("Preprocessor saved")

            df_trans = transform_data(
                transformer=preprocessor,
                data=df
            )
            logger.info("Train data transformed")

            save_dataframe(
                dataframe=df_trans,
                save_path=save_data_path / filename
            )
            logger.info("Transformed train data saved")

        elif filename == 'val.csv':
            df = read_dataframe(complete_input_path)
            logger.info(f"Validation dataframe loaded with shape {df.shape}")

            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            logger.info("Preprocessor loaded for validation data")

            df_trans = transform_data(
                transformer=preprocessor,
                data=df
            )
            logger.info("Validation data transformed")

            save_dataframe(
                dataframe=df_trans,
                save_path=save_data_path / filename
            )
            logger.info("Transformed validation data saved")

        elif filename == 'test.csv':
            df = read_dataframe(complete_input_path)
            logger.info(f"Test dataframe loaded with shape {df.shape}")

            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            logger.info("Preprocessor loaded for test data")

            X_trans = transform_data(
                transformer=preprocessor,
                data=df
            )
            logger.info("Test data transformed")

            save_dataframe(
                dataframe=X_trans,
                save_path=save_data_path / filename
            )
            logger.info("Transformed test data saved")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    app()
