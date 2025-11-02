import pandas as pd
import yaml

class DataValidationTrainingPipeline:
    def __init__(self):
        self.data_path = "artifacts/heart_raw.csv"
        self.schema_path = "schema.yaml"

    def main(self):
        # Load data
        df = pd.read_csv(self.data_path)

        # Load schema
        with open(self.schema_path, "r") as f:
            schema = yaml.safe_load(f)["COLUMNS"]

        # Check column names
        missing_cols = [col for col in schema if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in schema]

        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
        if extra_cols:
            print(f"⚠️ Extra columns: {extra_cols}")
        if not missing_cols and not extra_cols:
            print("✅ Column names match schema.")

        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print("❌ Missing values detected:")
            print(null_counts[null_counts > 0])
        else:
            print("✅ No missing values found.")
