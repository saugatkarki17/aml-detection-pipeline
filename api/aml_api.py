from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../src/xgb_AML_model_advanced.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]
feature_names = bundle["features"]

class AMLInput(BaseModel):
    amount_paid: float
    amount_received: float
    txn_count: int
    total_sent: float
    hour_of_day: int
    day_of_week: int
    payment_format: str
    payment_currency: str
    receiving_currency: str

# All categories expected (from your training)
PAYMENT_FORMATS = ["Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = [
    "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro", "Mexican Peso", "Ruble",
    "Rupee", "Saudi Riyal", "Shekel", "Swiss Franc", "UK Pound", "US Dollar", "Yen", "Yuan"
]

@app.post("/predict")
def predict(input_data: AMLInput):
    try:
        # Manual encoding(one-hot)
        def one_hot_encode(value, prefix, choices):
            return {f"{prefix}_{c}": int(c == value) for c in choices}

        row = {
            "amount_paid": input_data.amount_paid,
            "amount_received": input_data.amount_received,
            "amount_diff": abs(input_data.amount_paid - input_data.amount_received),
            "ratio_received_paid": input_data.amount_received / (input_data.amount_paid + 1e-5),
            "txn_count": input_data.txn_count,
            "total_sent": input_data.total_sent,
            "avg_txn_amount": input_data.total_sent / (input_data.txn_count + 1e-5),
            "z_score_paid": (input_data.amount_paid - 1000) / 300,
            "hour_of_day": input_data.hour_of_day,
            "day_of_week": input_data.day_of_week,

            # Using dummy values for missing graph-based features
            "unique_receivers": 5,
            "in_degree": 3,
            "out_degree": 2,
            "pagerank": 0.001
        }

        row.update(one_hot_encode(input_data.payment_format, "payment_format", PAYMENT_FORMATS))
        row.update(one_hot_encode(input_data.payment_currency, "payment_currency", CURRENCIES))
        row.update(one_hot_encode(input_data.receiving_currency, "receiving_currency", CURRENCIES))

        df = pd.DataFrame([row])

        # Ensure Match btn model's feature list exactly
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0  # If column is missing, fill with 0

        df = df[feature_names]

        proba = model.predict_proba(df)[0][1]
        prediction = int(proba >= threshold)

        # Risk levels (based on probability thresholds) 
        if proba >= 0.35:
            risk_level = "High"
        elif proba >= 0.10:
            risk_level = "Medium"
        elif proba >= 0.02:
            risk_level = "Low"
        else:
            risk_level = "Very Low"

        return {
        "prediction": int(prediction),              
        "probability": round(float(proba), 4),      
        "threshold": round(float(threshold), 3) ,
        "risk_level": risk_level   
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
