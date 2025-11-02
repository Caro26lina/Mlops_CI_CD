import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        self.data_path = "artifacts/heart_transformed.csv"
        self.model_path = "artifacts/model.pkl"
        self.metrics_path = "artifacts/metrics.json"

    def main(self):
        # Load data and model
        df = pd.read_csv(self.data_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predict
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # Save metrics
        os.makedirs("artifacts", exist_ok=True)
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print("✅ Model evaluation complete. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
