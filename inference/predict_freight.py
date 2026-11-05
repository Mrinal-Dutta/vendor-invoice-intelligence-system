
import joblib
import pandas as pd
from pathlib import Path

# Model path
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "freight_model.pkl"


def load_model(model_path=MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    model = joblib.load(model_path)
    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Freight"] = (
        model.predict(input_df).round(2)
    )

    return input_df


if __name__ == "__main__":

    sample_data = {
        "Quantity": [15, 25],
        "Dollars": [18500, 9000]
    }

    prediction = predict_freight_cost(sample_data)

    print(prediction)
