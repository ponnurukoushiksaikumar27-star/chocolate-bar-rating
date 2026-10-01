import joblib
import pandas as pd

def predict_rating(
    company,
    bean_origin,
    ref,
    review_date,
    cocoa_percent,
    company_location,
    bean_type,
    broad_bean_origin
):
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/random_forest_model.pkl")

    row = pd.DataFrame([{
        "company": company,
        "bean_origin": bean_origin,
        "ref": ref,
        "review_date": review_date,
        "cocoa_percent": cocoa_percent,
        "company_location": company_location,
        "bean_type": bean_type,
        "broad_bean_origin": broad_bean_origin
    }])
    X = preprocessor.transform(row)
    prediction = float(model.predict(X)[0])
    return max(0.0, min(5.0, prediction))

if __name__ == "__main__":
    # Example input. Replace these values with your own cacao/chocolate details.
    rating = predict_rating(
        company="A. Morin",
        bean_origin="Kpime",
        ref=1676,
        review_date=2015,
        cocoa_percent=70,
        company_location="France",
        bean_type="Trinitario",
        broad_bean_origin="Togo"
    )
    print(f"Predicted cacao/chocolate rating: {rating:.2f} / 5")
