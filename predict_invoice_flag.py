
import joblib
import pandas as pd
from pathlib import Path

# Project folder path
BASE_DIR = Path(__file__).resolve().parent.parent

# Model and scaler paths
MODEL_PATH = BASE_DIR / "models" / "invoice_flagging_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "invoice_scaler.pkl"


def load_model():
    """
    Load trained invoice classification model.
    """
    model = joblib.load(MODEL_PATH)
    return model


def load_scaler():
    """
    Load trained StandardScaler.
    """
    scaler = joblib.load(SCALER_PATH)
    return scaler


def predict_invoice_flag(input_data):
    """
    Predict whether an invoice should be flagged.
    """

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    # Match the exact feature order used during training
    feature_columns = list(scaler.feature_names_in_)
    input_df = input_df[feature_columns]

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Predict invoice flag
    prediction = model.predict(input_scaled)

    input_df["Predicted_Flag"] = prediction

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [100],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_brands": [5],
        "days_po_to_invoice": [10],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)
