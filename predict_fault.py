"""
Use the trained model to predict whether a new solar PV condition is normal or faulty.
"""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "solar_fault_model.pkl"


def predict_new_condition() -> None:
    """Predict fault status for one new solar PV operating condition."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run this command first: python src/train_model.py"
        )

    model = joblib.load(MODEL_PATH)

    # Change these values to test different system conditions
    new_data = pd.DataFrame(
        [
            {
                "irradiance": 850,
                "temperature": 65,
                "voltage": 170,
                "current": 7.5,
                "power": 1083.75,
                "inverter_efficiency": 72,
            }
        ]
    )

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0]

    if prediction == 1:
        result = "FAULT DETECTED"
    else:
        result = "NORMAL CONDITION"

    print("Input condition:")
    print(new_data)
    print("\nPrediction:", result)
    print(f"Normal probability: {probability[0]:.2f}")
    print(f"Fault probability: {probability[1]:.2f}")


if __name__ == "__main__":
    predict_new_condition()
