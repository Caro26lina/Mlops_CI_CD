import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataTransformationTrainingPipeline:
    def __init__(self):
        self.input_path = "artifacts/heart_raw.csv"
        self.output_path = "artifacts/heart_transformed.csv"

    def main(self):
        # Load data
        df = pd.read_csv(self.input_path)

        # Separate features and target
        X = df.drop("target", axis=1)
        y = df["target"]

        # Normalize numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Combine back into DataFrame
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        transformed_df = pd.concat([X_scaled_df, y], axis=1)

        # Save transformed data
        os.makedirs("artifacts", exist_ok=True)
        transformed_df.to_csv(self.output_path, index=False)
        print(f"✅ Transformed data saved to {self.output_path}")
