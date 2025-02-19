import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import joblib
import pandas as pd
from src.utils import preprocess_data


def predict(input_data):
    model = joblib.load("models/linear_regression.pkl")
    processed_data, _, _, _, _ = preprocess_data(pd.DataFrame([input_data]))
    prediction = model.predict(processed_data)
    return prediction[0]


if __name__ == "__main__":
    sample_input = {
        "MedInc": 3.5,
        "HouseAge": 25.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1500,
        "AveOccup": 3.0,
        "Latitude": 34.2,
        "Longitude": -118.4,
    }
    result = predict(sample_input)
    print(f"Predicted House Price: {result:.2f}")
