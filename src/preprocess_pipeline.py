import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

RAW_PATH = "data/raw/flavors_of_cacao.csv"

def clean_columns(df):
    df = df.copy()
    df.columns = [c.replace("\xa0", " ").replace("\n", " ").strip() for c in df.columns]
    # Normalize the dataset column names
    rename = {
        "Company (Maker-if known)": "company",
        "Specific Bean Origin or Bar Name": "bean_origin",
        "REF": "ref",
        "Review Date": "review_date",
        "Cocoa Percent": "cocoa_percent",
        "Company Location": "company_location",
        "Rating": "rating",
        "Bean Type": "bean_type",
        "Broad Bean Origin": "broad_bean_origin",
    }
    df = df.rename(columns=rename)

    # Treat whitespace-only values as missing.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True).astype("string").str.strip()

    df["cocoa_percent"] = pd.to_numeric(
        df["cocoa_percent"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["ref"] = pd.to_numeric(df["ref"], errors="coerce")
    df["review_date"] = pd.to_numeric(df["review_date"], errors="coerce")
    return df

def run_preprocessing():
    print("[INFO] Starting Cacao preprocessing pipeline...")
    df = clean_columns(pd.read_csv(RAW_PATH))
    df = df.dropna(subset=["rating"])

    X = df.drop(columns=["rating"])
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=float))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=float))
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    feature_names = list(preprocessor.get_feature_names_out())
    metadata = {
        "dataset_name": "Flavors of Cacao",
        "problem_type": "regression",
        "target": "rating",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "transformed_feature_count": len(feature_names)
    }
    with open("data/processed/dataset_metadata.json","w") as f:
        json.dump(metadata,f,indent=4)

    print("[SUCCESS] Preprocessing completed.")
    print(f"[INFO] Train shape: {X_train_final.shape}; Test shape: {X_test_final.shape}")

if __name__ == "__main__":
    run_preprocessing()
