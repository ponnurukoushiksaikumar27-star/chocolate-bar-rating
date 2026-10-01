# Cacao Flavor Rating Prediction

A machine-learning project based on the **Flavors of Cacao** dataset.

## What this project does
It follows the same overall workflow as the supplied Churn Prediction project:

1. Validate the raw dataset
2. Clean and preprocess data
3. Split into training/testing sets
4. Encode categorical features and scale numeric features
5. Train a Random Forest model
6. Evaluate the model
7. Perform error analysis
8. Validate reproducibility
9. Optionally track the experiment with MLflow
10. Save model/preprocessing artifacts

### Target
`Rating` — the chocolate rating in the dataset. This is a **regression** problem because Rating is numeric.

### Main input features
- Company / maker
- Specific bean origin or bar name
- REF
- Review date
- Cocoa percentage
- Company location
- Bean type
- Broad bean origin

## Project structure
```text
Cacao_Flavor_Rating_Prediction/
├── data/
│   ├── raw/flavors_of_cacao.csv
│   └── processed/
├── models/
├── artifacts/
├── outputs/
├── src/
├── pipelines/
├── notebooks/
├── configs/
├── requirements.txt
└── README.md
```

## Windows setup

Open Command Prompt/PowerShell inside this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt or run:
```bash
.venv\Scripts\activate.bat
```

## Run the complete local ML pipeline

```bash
python pipelines/run_pipeline.py
```

Or run each stage separately:

```bash
python src/validate_data.py
python src/preprocess_pipeline.py
python src/validate_outputs.py
python src/train.py
python src/evaluate.py
python src/validate_reproducibility.py
```

## MLflow experiment tracking (optional)

After preprocessing:

```bash
python src/train_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the local MLflow URL shown in the terminal.

## Important
Do not type the terminal prompt itself. For example, if the terminal shows:

```text
(.venv) C:\Cacao_Flavor_Rating_Prediction>
```

type only the command after `>`.

## Output
The main trained model is saved as:
`models/random_forest_model.pkl`

The preprocessing pipeline is saved as:
`models/preprocessor.pkl`

Evaluation metrics and plots are saved under:
`artifacts/`

Error-analysis CSV files are saved under:
`outputs/`

## Verified run in this environment

The complete pipeline was executed successfully.

Example test-set metrics from the Random Forest model:
- MAE: 0.3575
- RMSE: 0.4694
- R²: 0.0470

These values are a measured result for this dataset/split, not a guarantee on future data.

The project also passed preprocessing-output validation and reproducibility validation.

## Simple prediction test

```bash
python src/predict.py
```

This loads the saved preprocessing pipeline and trained model and predicts a rating on a 0–5 scale for the example input.
