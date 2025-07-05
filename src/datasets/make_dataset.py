from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from helper import convert_to_minutes, clean_data

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "flight_price.csv",
):
    """
    Process the dataset and split into train, validation, and test CSVs.
    """
    logger.info("Processing dataset...")
    data = pd.read_csv(input_path)
    logger.info(f"Dataset loaded with shape {data.shape}.")

    cleaned_data = clean_data(data)
    logger.info(f"Data cleaned. Shape after cleaning: {cleaned_data.shape}")

    # First split into train and temp (train 80%, temp 20%)
    train_df, temp_df = train_test_split(cleaned_data, test_size=0.2, random_state=42)

    # Split temp into validation and test (each 10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Validation shape: {val_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    # Ensure output dir exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save files with simple names
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train set saved to {train_path}")
    logger.info(f"Validation set saved to {val_path}")
    logger.info(f"Test set saved to {test_path}")
    logger.success("All splits completed successfully!")

if __name__ == "__main__":
    app()
