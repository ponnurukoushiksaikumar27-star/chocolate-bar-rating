import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

def run_training():
    print("[INFO] Starting Random Forest training...")
    X_train = np.load("data/processed/X_train_final.npy")
    y_train = np.load("data/processed/y_train.npy")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_model.pkl")
    joblib.dump(model, "models/random_forest_baseline.pkl")
    print("[SUCCESS] Model saved to models/random_forest_model.pkl")

if __name__ == "__main__":
    run_training()
