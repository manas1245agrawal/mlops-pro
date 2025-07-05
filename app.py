from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

current_file_path = Path(__file__).parent

model_path = current_file_path / "models" / "models" / "xgbreg.joblib"
preprocessor_path = model_path.parent.parent / "transformers" / "preprocessor.joblib"

# model = joblib.load(model_path)
# preprocessor = joblib.load(preprocessor_path)

# model_pipe = Pipeline(steps=[
#     ('preprocess', preprocessor),
#     ('regressor', model)
# ])

class RequestBody(BaseModel):
    air__airline_Indigo: float
    air__airline_Jet_Airways: float
    air__airline_Other: float
    doj__date_of_journey_week: float
    doj__date_of_journey_day_of_year: float
    location__source: str
    location__destination: str
    dur__duration_rbf_25: float
    dur__duration_cat: str
    dur__duration_over_1000: float
    dur__duration: float
    stops__total_stops: float
    stops__is_direct_flight: float

@app.get('/')
def home():
    return "Welcome to taxi price prediction app"

@app.post('/predictions')
def do_predictions(requestBody: RequestBody):
    X_test = pd.DataFrame(
        data={
            'air__airline_Indigo': requestBody.air__airline_Indigo,
            'air__airline_Jet Airways': requestBody.air__airline_Jet_Airways,
            'air__airline_Other': requestBody.air__airline_Other,
            'doj__date_of_journey_week': requestBody.doj__date_of_journey_week,
            'doj__date_of_journey_day_of_year': requestBody.doj__date_of_journey_day_of_year,
            'location__source': requestBody.location__source,
            'location__destination': requestBody.location__destination,
            'dur__duration_rbf_25': requestBody.dur__duration_rbf_25,
            'dur__duration_cat': requestBody.dur__duration_cat,
            'dur__duration_over_1000': requestBody.dur__duration_over_1000,
            'dur__duration': requestBody.dur__duration,
            'stops__total_stops': requestBody.stops__total_stops,
            'stops__is_direct_flight': requestBody.stops__is_direct_flight
        },
        index=[0]
    )
    predictions = [[0]]
    # predictions = model_pipe.predict(X_test).reshape(-1, 1)

    return f"Trip duration for the trip is {predictions[0][0]:.2f} minutes"

if __name__ == "__main__":
    uvicorn.run(app="app:app",
                host="0.0.0.0",
                port=5000)
