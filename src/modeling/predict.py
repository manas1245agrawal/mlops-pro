from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import joblib
import pandas as pd
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from sklearn.metrics import r2_score
app = typer.Typer()

TARGET = 'price'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df
    
    
def make_X_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()
    
    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]
    
    return X, y

def get_predictions(model,X:pd.DataFrame):
    # get predictions on data
    y_pred = model.predict(X)
    
    return y_pred
    
def calculate_r2_score(y_actual,y_predicted):
    score = r2_score(y_actual,y_predicted)
    return score

@app.command()
def main():
    features_path: Path = PROCESSED_DATA_DIR / 'final' / "val.csv"
    model_path: Path = MODELS_DIR / "model" / "model.pkl"
    data = load_dataframe(features_path)
    X_test, y_test = make_X_y(dataframe=data,target_column=TARGET)
    model = joblib.load(model_path)
    y_pred = get_predictions(model=model,X=X_test)
    score = calculate_r2_score(y_actual=y_test,y_predicted=y_pred)



if __name__ == "__main__":
    app()
