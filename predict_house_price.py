import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def train_area_price_model(file_path="housing_data.csv"):
    """
    Trains a Linear Regression model predicting house price based on Area (Square Feet).
    """
    if not os.path.exists(file_path):
        # Generate sample data if missing
        np.random.seed(42)
        sq_ft = np.random.randint(600, 4000, size=250)
        # Price = $50,000 + ($160 * sq_ft) + noise
        price = 50000 + (sq_ft * 160) + np.random.normal(0, 20000, size=250)
        df = pd.DataFrame({"Square_Feet": sq_ft, "Price": np.round(price, 2)})
        df.to_csv(file_path, index=False)
        
    df = pd.read_csv(file_path)
    
    # Select area (Square_Feet) and price columns
    area_col = "Square_Feet" if "Square_Feet" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    price_col = "Price" if "Price" in df.columns else df.select_dtypes(include=[np.number]).columns[-1]
    
    X = df[[area_col]]
    y = df[price_col]
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    return model, area_col, price_col

def predict_house_price(area_value, model, feature_name="Square_Feet"):
    """Predicts house price given an area input in square feet."""
    area_df = pd.DataFrame([[float(area_value)]], columns=[feature_name])
    price = model.predict(area_df)[0]
    return price

if __name__ == "__main__":
    model, area_col, price_col = train_area_price_model()
    
    # Get area from command line argument if provided, else default to 1500 sq ft
    if len(sys.argv) > 1:
        try:
            area_input = float(sys.argv[1])
        except ValueError:
            print("Error: Area input must be a valid number.")
            sys.exit(1)
    else:
        try:
            val = input("Enter house area in square feet (default 1500): ").strip()
            area_input = float(val) if val else 1500.0
        except (ValueError, EOFError):
            area_input = 1500.0
            
    predicted_price = predict_house_price(area_input, model, feature_name=area_col)
    
    print("\n" + "="*45)
    print("        HOUSE PRICE PREDICTION RESULT        ")
    print("="*45)
    print(f" Input Area          : {area_input:,.2f} sq ft")
    print(f" Predicted Price     : ${predicted_price:,.2f}")
    print("="*45)
