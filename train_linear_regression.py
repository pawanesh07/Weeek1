import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ensure_sample_dataset(file_path="housing_data.csv", n_samples=200):
    """Generates a synthetic housing dataset if none exists."""
    if not os.path.exists(file_path):
        np.random.seed(42)
        sq_ft = np.random.randint(800, 3500, size=n_samples)
        bedrooms = np.random.randint(1, 6, size=n_samples)
        age = np.random.randint(1, 30, size=n_samples)
        
        # Linear relationship with added noise
        price = 50000 + (sq_ft * 150) + (bedrooms * 10000) - (age * 1200) + np.random.normal(0, 15000, size=n_samples)
        
        df = pd.DataFrame({
            "Square_Feet": sq_ft,
            "Bedrooms": bedrooms,
            "Age": age,
            "Price": np.round(price, 2)
        })
        df.to_csv(file_path, index=False)
        print(f"Generated synthetic regression dataset: '{file_path}' ({n_samples} rows)\n")

def train_and_evaluate(file_path, target_col="Price"):
    """Loads dataset, trains Linear Regression model, and checks accuracy."""
    print(f"1. Loading dataset from '{file_path}'...")
    df = pd.read_csv(file_path)
    
    # Filter numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        # Use the last numeric column as target if target_col doesn't exist
        target_col = numeric_df.columns[-1]
        
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    
    print(f"Features: {list(X.columns)}")
    print(f"Target Variable: '{target_col}'\n")

    # Split into 80% Training set and 20% Testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression model
    print("2. Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate Evaluation Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n" + "="*50)
    print("         MODEL ACCURACY & EVALUATION METRICS       ")
    print("="*50)
    print(f" R^2 Score (Accuracy)       : {r2:.4f} ({r2 * 100:.2f}%)")
    print(f" Mean Absolute Error (MAE)  : {mae:,.2f}")
    print(f" Root Mean Squared Error    : {rmse:,.2f}")
    print("="*50)

    # Model parameters
    print("\n--- Model Coefficients & Intercept ---")
    print(f"Intercept: {model.intercept_:.2f}")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"Coefficient [{feature}]: {coef:.2f}")

    # First 5 predictions comparison
    print("\n--- First 5 Predictions (Actual vs Predicted) ---")
    comparison = pd.DataFrame({
        "Actual": y_test.values[:5],
        "Predicted": np.round(y_pred[:5], 2),
        "Difference": np.round(y_test.values[:5] - y_pred[:5], 2)
    })
    print(comparison.to_string(index=False))

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "housing_data.csv"
    
    # Ensure sample dataset exists if using default
    if csv_file == "housing_data.csv":
        ensure_sample_dataset(csv_file)
        
    train_and_evaluate(csv_file)
