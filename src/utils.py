# src/utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath="data/housing.csv"):
    # """Load dataset từ file CSV."""
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df, target_column="MedHouseVal"):  # Đảm bảo cột target đúng
    X = df.drop(columns=[target_column])  # Xóa cột target khỏi features
    y = df[target_column]
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

