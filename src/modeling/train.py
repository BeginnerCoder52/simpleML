import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sklearn.datasets import fetch_california_housing
import pandas as pd


def train_model():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target  # Cột mục tiêu chính xác

    from src.utils import preprocess_data

    X_train, X_test, y_train, y_test, _ = preprocess_data(
        df, target_column="MedHouseVal"
    )

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import joblib

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    joblib.dump(model, "models/linear_regression.pkl")
    print("Model saved to models/linear_regression.pkl")


if __name__ == "__main__":
    train_model()
