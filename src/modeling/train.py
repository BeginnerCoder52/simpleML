# src/train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.utils import load_data, preprocess_data


def train_model():
    # Sử dụng dataset mặc định của sklearn thay vì housing.csv
    from sklearn.datasets import load_boston

    data = load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["PRICE"] = data.target

    # Tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, target_column="PRICE")

    # Huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Lưu mô hình
    import joblib

    joblib.dump(model, "models/linear_regression.pkl")
    print("Model saved to models/linear_regression.pkl")


if __name__ == "__main__":
    train_model()
