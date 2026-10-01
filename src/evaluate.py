import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_evaluation():
    print("[INFO] Starting model evaluation...")
    X_test = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load("models/random_forest_model.pkl")

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2_score": float(r2_score(y_test, y_pred))
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/production_model_report.json","w") as f:
        json.dump({"problem_type":"regression","target":"Rating","metrics":metrics},f,indent=4)

    plt.figure(figsize=(7,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    low, high = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([low, high],[low, high], linestyle="--")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Actual vs Predicted Cacao Rating")
    plt.tight_layout()
    plt.savefig("artifacts/actual_vs_predicted.png", dpi=150)
    plt.close()

    errors = pd.DataFrame({
        "actual_rating": y_test,
        "predicted_rating": y_pred,
        "absolute_error": np.abs(y_test-y_pred)
    }).sort_values("absolute_error", ascending=False)
    errors.to_csv("outputs/prediction_errors.csv", index=False)

    print("\n--- Model Evaluation Report ---")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"R²   : {metrics['r2_score']:.4f}")
    print("-------------------------------")
    return metrics

if __name__ == "__main__":
    run_evaluation()
