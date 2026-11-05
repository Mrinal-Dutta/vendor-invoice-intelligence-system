
import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DB_PATH = PROJECT_DIR / "data" / "inventory.db"
MODEL_DIR = PROJECT_DIR / "models"


def main():

    # Create models folder if it does not exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(str(DB_PATH))

    # Prepare features
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)

    dt_model = train_decision_tree(X_train, y_train)

    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    results = []

    results.append(
        evaluate_model(
            lr_model, X_test, y_test, "Linear Regression"
        )
    )

    results.append(
        evaluate_model(
            dt_model, X_test, y_test, "Decision Tree Regression"
        )
    )

    results.append(
        evaluate_model(
            rf_model, X_test, y_test, "Random Forest Regression"
        )
    )

    # Select best model based on lowest MAE
    best_model_info = min(results, key=lambda x: x["mae"])

    best_model_name = best_model_info["model_name"]

    trained_models = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }

    best_model = trained_models[best_model_name]

    # Save best model
    model_path = MODEL_DIR / "freight_model.pkl"

    joblib.dump(best_model, model_path)

    print("\nBest Model:", best_model_name)
    print("Best MAE:", round(best_model_info["mae"], 2))
    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()
