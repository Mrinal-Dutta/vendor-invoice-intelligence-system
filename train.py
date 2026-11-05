
import joblib
from pathlib import Path

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features
)

from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_brands",
    "total_item_quantity",
    "days_po_to_invoice",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_invoice_data()

    # Create invoice risk labels
    df = apply_labels(df)

    # Remove rows with missing required values
    df = df.dropna(subset=FEATURES + [TARGET])

    # Prepare training and testing data
    X_train, X_test, y_train, y_test = split_data(
        df, FEATURES, TARGET
    )

    # Scale features
    scaler_path = MODEL_DIR / "invoice_scaler.pkl"

    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        scaler_path
    )

    # Train Random Forest with GridSearchCV
    grid_search = train_random_forest(
        X_train_scaled,
        y_train
    )

    # Evaluate best model
    best_model = grid_search.best_estimator_

    evaluate_classifier(
        best_model,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save model
    model_path = MODEL_DIR / "invoice_flagging_model.pkl"

    joblib.dump(best_model, model_path)

    print("\nBest Parameters:", grid_search.best_params_)
    print("Model saved at:", model_path)
    print("Scaler saved at:", scaler_path)


if __name__ == "__main__":
    main()
