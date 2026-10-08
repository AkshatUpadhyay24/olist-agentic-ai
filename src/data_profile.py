from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# 1. Find our raw data folder
# ---------------------------------------------------------

DATA_DIR = Path("data/raw")


# ---------------------------------------------------------
# 2. Find all CSV files
# ---------------------------------------------------------

csv_files = list(DATA_DIR.glob("*.csv"))

print("\n" + "=" * 70)
print("OLLIST DATASET PROFILE")
print("=" * 70)

print(f"\nTotal CSV files found: {len(csv_files)}")


# ---------------------------------------------------------
# 3. Read every CSV and display basic information
# ---------------------------------------------------------

for file in csv_files:

    print("\n" + "-" * 70)
    print(f"FILE: {file.name}")
    print("-" * 70)

    df = pd.read_csv(file)

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nData types:")
    print(df.dtypes)