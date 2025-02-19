# src/predict.py
import joblib
import numpy as np
import pandas as pd
from src.utils import preprocess_data


def predict(input_data):
    """Dự đoán giá nhà dựa trên dữ liệu đầu vào."""
    model = joblib.load("models/linear_regression.pkl")
    processed_data, _, _, _, _ = preprocess_data(pd.DataFrame([input_data]))
    prediction = model.predict(processed_data)
    return prediction[0]


if __name__ == "__main__":
    # Ví dụ dữ liệu đầu vào (cần thay đổi phù hợp với dataset cụ thể)
    sample_input = {
        "CRIM": 0.1,
        "ZN": 25.0,
        "INDUS": 5.0,
        "CHAS": 0,
        "NOX": 0.5,
        "RM": 6.0,
        "AGE": 30.0,
        "DIS": 4.0,
        "RAD": 4,
        "TAX": 300,
        "PTRATIO": 15.0,
        "B": 390.0,
        "LSTAT": 5.0,
    }
    result = predict(sample_input)
    print(f"Predicted Price: {result:.2f}")
