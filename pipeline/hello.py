import os
import pandas as pd

# List all files in the cleaned_data folder
data_folder = "cleaned_data"
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Load all datasets into a dictionary
datasets = {file: pd.read_csv(os.path.join(data_folder, file)) for file in files}

# Define checks
def run_checks(df):
    checks = {
        "Shape": df.shape,
        "Missing values": df.isnull().sum().sum(),
        "Duplicate rows": df.duplicated().sum(),
        "Constant columns": [col for col in df.columns if df[col].nunique() == 1],
        "All numeric columns": all([pd.api.types.is_numeric_dtype(df[col]) for col in df.select_dtypes(include=['number']).columns]),
        "All string columns": all([pd.api.types.is_string_dtype(df[col]) for col in df.select_dtypes(include=['object']).columns]),
        "Negative values (numeric columns)": {col: (df[col] < 0).sum() for col in df.select_dtypes(include=['number']).columns},
        "Outliers (z-score > 3)": {col: ((df[col] - df[col].mean()).abs() > 3*df[col].std()).sum() for col in df.select_dtypes(include=['number']).columns},
        "Unique values per column": {col: df[col].nunique() for col in df.columns},
        "Whitespace in string columns": {col: df[col].astype(str).str.contains(r'^\s+|\s+$').sum() for col in df.select_dtypes(include=['object']).columns},
    }
    return checks

# Run checks for each dataset
results = {file: run_checks(df) for file, df in datasets.items()}

# Display results
for file, checks in results.items():
    print(f"Checks for {file}:")
    for check, result in checks.items():
        print(f"  {check}: {result}")
    print("-" * 40)