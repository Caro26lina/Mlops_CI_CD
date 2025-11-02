import pandas as pd
import os

class DataIngestionTrainingPipeline:
    def __init__(self):
        self.output_path = "artifacts/heart_raw.csv"

    def main(self):
        # Load from local CSV
        df = pd.read_csv("data/heart.csv")  # Make sure this path matches your file

        os.makedirs("artifacts", exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"✅ Data ingested and saved to {self.output_path}")
