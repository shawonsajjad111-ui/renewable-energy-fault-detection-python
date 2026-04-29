"""
Train a machine learning model to detect solar PV system faults.

The script:
1. Loads the generated CSV dataset
2. Splits data into training and testing sets
3. Trains a Random Forest Classifier
4. Evaluates model performance
5. Saves plots, metrics, predictions, and trained model
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "solar_fault_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "solar_fault_model.pkl"


def load_dataset() -> pd.DataFrame:
    """Load the solar fault dataset from CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run this command first: python src/generate_data.py"
        )
    return pd.read_csv(DATA_PATH)


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, pd.DataFrame, pd.Series, pd.Series]:
    """Train Random Forest model and return model plus test data."""
    features = [
        "irradiance",
        "temperature",
        "voltage",
        "current",
        "power",
        "inverter_efficiency",
    ]

    x = df[features]
    y = df["fault"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=8,
    )
    model.fit(x_train, y_train)

    return model, x_test, y_test, x


def save_evaluation_outputs(model, x_test, y_test, all_features) -> None:
    """Save model evaluation files in the outputs folder."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Fault"])

    metrics_text = f"Model Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}"
    (OUTPUT_DIR / "model_metrics.txt").write_text(metrics_text, encoding="utf-8")

    predictions = x_test.copy()
    predictions["actual_fault"] = y_test.values
    predictions["predicted_fault"] = y_pred
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fault"])
    display.plot()
    plt.title("Confusion Matrix - Solar PV Fault Detection")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Feature importance plot
    importance = pd.Series(model.feature_importances_, index=all_features.columns)
    importance = importance.sort_values(ascending=True)
    importance.plot(kind="barh")
    plt.title("Feature Importance for Fault Detection")
    plt.xlabel("Importance Score")
    plt.savefig(OUTPUT_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close()

    # Power vs irradiance chart
    df_plot = pd.read_csv(DATA_PATH)
    plt.scatter(df_plot["irradiance"], df_plot["power"], c=df_plot["fault"], alpha=0.7)
    plt.title("Power vs Irradiance: Normal vs Fault Condition")
    plt.xlabel("Irradiance (W/m2)")
    plt.ylabel("Power (W)")
    plt.savefig(OUTPUT_DIR / "power_vs_irradiance.png", bbox_inches="tight")
    plt.close()

    joblib.dump(model, MODEL_PATH)

    print(metrics_text)
    print(f"Outputs saved inside: {OUTPUT_DIR}")


if __name__ == "__main__":
    dataset = load_dataset()
    trained_model, x_test_data, y_test_data, feature_data = train_model(dataset)
    save_evaluation_outputs(trained_model, x_test_data, y_test_data, feature_data)
