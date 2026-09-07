import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def ensure_sample_dataset(file_path="sample_data.csv"):
    """Generates a sample dataset if none exists."""
    if not os.path.exists(file_path):
        data = {
            "ID": list(range(1, 101)),
            "Feature_1": [i * 1.5 for i in range(1, 101)],
            "Feature_2": [i * 2.3 for i in range(1, 101)],
            "Category": ["A", "B", "C", "D"] * 25,
            "Target": [0 if i % 2 == 0 else 1 for i in range(1, 101)]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"Generated sample dataset: '{file_path}' (100 rows)\n")

def split_dataset(file_path, target_column="Target", test_size=0.2, random_state=42):
    
    print(f"Loading dataset from: '{file_path}'")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape} (rows, columns)\n")

    if target_column in df.columns:
        # Separate features (X) and target variable (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"=== Split Summary (test_size={test_size*100:.0f}%) ===")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape : {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape : {y_test.shape}")
        
        print("\n--- First 5 rows of X_train ---")
        print(X_train.head())
        
        return X_train, X_test, y_train, y_test
    else:
        # Split full DataFrame into train_df and test_df
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        print(f"=== DataFrame Split Summary (test_size={test_size*100:.0f}%) ===")
        print(f"Train DataFrame shape: {train_df.shape}")
        print(f"Test DataFrame shape : {test_df.shape}")
        
        print("\n--- First 5 rows of Train DataFrame ---")
        print(train_df.head())
        
        return train_df, test_df

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    
    ensure_sample_dataset(csv_file)
    
    split_dataset(csv_file, target_column="Target", test_size=0.2, random_state=42)
