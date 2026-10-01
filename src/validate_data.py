import os
import pandas as pd

RAW_PATH = "data/raw/flavors_of_cacao.csv"

EXPECTED_COLUMNS = [
    "Company\xa0\n(Maker-if known)",
    "Specific Bean Origin\nor Bar Name",
    "REF",
    "Review\nDate",
    "Cocoa\nPercent",
    "Company\nLocation",
    "Rating",
    "Bean\nType",
    "Broad Bean\nOrigin",
]

def validate_dataset(df):
    errors = []
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if "Rating" in df.columns:
        if pd.to_numeric(df["Rating"], errors="coerce").isna().any():
            errors.append("Rating contains non-numeric values.")
        if not pd.to_numeric(df["Rating"], errors="coerce").between(0, 5).all():
            errors.append("Rating contains values outside 0-5.")

    if "Cocoa\nPercent" in df.columns:
        cocoa = pd.to_numeric(df["Cocoa\nPercent"].astype(str).str.replace("%","",regex=False), errors="coerce")
        if cocoa.isna().any():
            errors.append("Cocoa percentage contains non-numeric values after % removal.")
        if not cocoa.between(0,100).all():
            errors.append("Cocoa percentage contains values outside 0-100.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": {str(k): int(v) for k,v in df.isna().sum().items()},
        "errors": errors
    }
    os.makedirs("artifacts", exist_ok=True)
    import json
    with open("artifacts/data_validation_report.json","w") as f:
        json.dump(report,f,indent=4)

    if errors:
        print("[ERROR] Dataset validation FAILED")
        for e in errors: print(" -", e)
        return False
    print(f"[SUCCESS] Dataset validation PASSED: {len(df)} rows, {len(df.columns)} columns")
    return True

if __name__ == "__main__":
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    if not validate_dataset(df):
        raise SystemExit(1)
