import os
import sys
import pandas as pd

def ensure_sample_csv(file_path):
    """Generates a sample CSV file if none exists."""
    if not os.path.exists(file_path):
        data = {
            "ID": list(range(1, 16)),
            "Name": [f"Item_{i}" for i in range(1, 16)],
            "Category": ["Electronics", "Clothing", "Books", "Home", "Garden"] * 3,
            "Price": [10.5 * i for i in range(1, 16)],
            "Stock": [50 + i * 2 for i in range(1, 16)]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"Created sample dataset at: '{file_path}'\n")

def load_and_display_csv(file_path):
    """Loads a CSV file using pandas and prints the first 10 rows."""
    try:
        # Load CSV using pandas
        df = pd.read_csv(file_path)
        
        # Display first 10 rows
        print(f"--- First 10 Rows of '{file_path}' ---")
        print(df.head(10))
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"Error loading CSV: {e}")

if __name__ == "__main__":
    # Use file path from command line argument, or default to 'sample_data.csv'
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    
    # Ensure sample CSV exists if using default
    if csv_file == "sample_data.csv":
        ensure_sample_csv(csv_file)
        
    load_and_display_csv(csv_file)
