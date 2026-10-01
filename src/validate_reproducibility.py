import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_predict():
    X_train=np.load("data/processed/X_train_final.npy")
    X_test=np.load("data/processed/X_test_final.npy")
    y_train=np.load("data/processed/y_train.npy")
    y_test=np.load("data/processed/y_test.npy")
    model=RandomForestRegressor(
        n_estimators=200,max_depth=15,min_samples_leaf=2,random_state=42,n_jobs=-1
    )
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    metrics={
        "mae":mean_absolute_error(y_test,pred),
        "rmse":np.sqrt(mean_squared_error(y_test,pred)),
        "r2_score":r2_score(y_test,pred)
    }
    return pred,metrics

def validate_reproducibility():
    p1,m1=train_and_predict()
    p2,m2=train_and_predict()
    predictions_match=np.allclose(p1,p2)
    metrics_match=all(np.isclose(m1[k],m2[k]) for k in m1)
    print("Predictions identical :",predictions_match)
    print("Metrics identical     :",metrics_match)
    if predictions_match and metrics_match:
        print("[SUCCESS] Reproducibility validation passed!")
    else:
        raise SystemExit("[ERROR] Reproducibility validation failed.")

if __name__=="__main__":
    validate_reproducibility()
