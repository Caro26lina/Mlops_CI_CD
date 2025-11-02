import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

class ModelTrainerTrainingPipeline:
    def __init__(self):
        self.data_path = "artifacts/heart_transformed.csv"
        self.model_path = "artifacts/model.pkl"

    def main(self):
        # Load transformed data
        df = pd.read_csv(self.data_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Start MLflow run
        mlflow.set_experiment("HeartDiseasePrediction")
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log to MLflow
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")

            # Save locally
            os.makedirs("artifacts", exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)

            print("✅ Model training complete.")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
