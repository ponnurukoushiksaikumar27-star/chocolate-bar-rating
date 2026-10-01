import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_track(run_name="RandomForest_Cacao_Baseline"):
    X_train=np.load("data/processed/X_train_final.npy")
    X_test=np.load("data/processed/X_test_final.npy")
    y_train=np.load("data/processed/y_train.npy")
    y_test=np.load("data/processed/y_test.npy")

    params={"n_estimators":200,"max_depth":15,"min_samples_leaf":2,"random_state":42}
    mlflow.set_experiment("Cacao_Flavor_Rating_Prediction")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family","RandomForestRegressor")
        mlflow.log_param("target","Rating")
        model=RandomForestRegressor(**params,n_jobs=-1)
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        metrics={
            "mae":mean_absolute_error(y_test,pred),
            "rmse":np.sqrt(mean_squared_error(y_test,pred)),
            "r2_score":r2_score(y_test,pred)
        }
        mlflow.log_metrics(metrics)
        if os.path.exists("data/processed/dataset_metadata.json"):
            mlflow.log_artifact("data/processed/dataset_metadata.json",artifact_path="metadata")
        if os.path.exists("models/preprocessor.pkl"):
            mlflow.log_artifact("models/preprocessor.pkl",artifact_path="preprocessing")
        mlflow.sklearn.log_model(model,name="cacao_rating_model")
        joblib.dump(model,"models/random_forest_model.pkl")
        print(f"[SUCCESS] MLflow run logged: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2_score']:.4f}")

if __name__=="__main__":
    train_and_track()
