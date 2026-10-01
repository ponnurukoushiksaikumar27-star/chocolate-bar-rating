import subprocess
import sys

def run():
    scripts=[
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py",
        "src/train.py",
        "src/evaluate.py",
        "src/validate_reproducibility.py",
    ]
    print("==============================================")
    print(" Cacao Flavor Rating Prediction - Full Pipeline")
    print("==============================================")
    for script in scripts:
        print(f"\n[INFO] Running {script}...")
        result=subprocess.run([sys.executable,script])
        if result.returncode != 0:
            raise SystemExit(f"[ERROR] Pipeline stopped at {script}")
    print("\n[SUCCESS] Complete pipeline finished successfully.")

if __name__=="__main__":
    run()
