import joblib
import sys
import pandas as pd
from yaml import safe_load
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from loguru import logger  # ✅ added loguru
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROJ_ROOT, DATA_DIR

TARGET = 'price'

def load_dataframe(path):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.debug(f"Data shape: {df.shape}")
    return df

def make_X_y(dataframe: pd.DataFrame, target_column: str):
    logger.info(f"Splitting dataframe into features and target '{target_column}'")
    df_copy = dataframe.copy()
    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]
    logger.debug(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def train_model(model, X_train, y_train):
    logger.info(f"Training model: {model.__class__.__name__}")
    model.fit(X_train, y_train)
    logger.success("Model training complete")
    return model

def save_model(model, save_path):
    logger.info(f"Saving model to {save_path}")
    joblib.dump(value=model, filename=save_path)
    logger.success("Model saved")

def main():
    logger.info("Training pipeline started")

    training_data_path = DATA_DIR / 'processed'/ 'train.csv'
    train_data = load_dataframe(training_data_path)

    X_train, y_train = make_X_y(dataframe=train_data, target_column=TARGET)

    regressor = XGBRegressor()
    # regressor = train_model(model=regressor, X_train=X_train, y_train=y_train)

    model_output_path = PROJ_ROOT / 'models' / 'models'
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    save_model(model=regressor, save_path=model_output_path / 'xgbreg.joblib')

    logger.success("Training pipeline completed")

if __name__ == "__main__":
    main()
