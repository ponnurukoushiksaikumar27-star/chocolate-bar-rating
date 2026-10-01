import json
import os
import numpy as np

def validate_preprocessing_outputs():
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    errors=[]
    if np.isnan(X_train).sum() or np.isnan(X_test).sum():
        errors.append("NaN values remain in transformed features.")
    if X_train.shape[1] != X_test.shape[1]:
        errors.append("Train/test feature dimensions do not match.")
    if len(y_train) != len(X_train):
        errors.append("Training feature/target row count mismatch.")
    if len(y_test) != len(X_test):
        errors.append("Testing feature/target row count mismatch.")

    report={
        "validation_status":"PASSED" if not errors else "FAILED",
        "X_train_shape":list(X_train.shape),
        "X_test_shape":list(X_test.shape),
        "y_train_shape":list(y_train.shape),
        "y_test_shape":list(y_test.shape),
        "errors":errors
    }
    os.makedirs("artifacts",exist_ok=True)
    with open("artifacts/preprocessing_summary_report.json","w") as f:
        json.dump(report,f,indent=4)
    if errors:
        print("[ERROR] Output validation failed.")
        for e in errors: print(" -",e)
        return False
    print("[SUCCESS] Preprocessing output validation passed.")
    return True

if __name__=="__main__":
    if not validate_preprocessing_outputs(): raise SystemExit(1)
