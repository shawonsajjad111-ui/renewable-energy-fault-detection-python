"""
Generate a synthetic solar PV fault detection dataset.

This script creates realistic sensor-style data for a solar PV system.
The data is suitable for beginner-level machine learning practice.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_SAMPLES = 1200


def generate_solar_fault_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Create synthetic solar PV system data with normal and faulty conditions."""
    np.random.seed(RANDOM_SEED)

    # Normal environmental and electrical behavior
    irradiance = np.random.uniform(150, 1000, n_samples)  # W/m2
    temperature = np.random.uniform(20, 55, n_samples)  # Celsius

    voltage = 220 + np.random.normal(0, 8, n_samples)
    current = (irradiance / 1000) * 18 + np.random.normal(0, 1.2, n_samples)
    inverter_efficiency = np.random.uniform(88, 98, n_samples)

    # Power formula: P = V * I * efficiency
    power = voltage * current * (inverter_efficiency / 100)

    # Start with all systems normal
    fault = np.zeros(n_samples, dtype=int)

    # Randomly select 22% of records as faulty
    fault_indices = np.random.choice(
        np.arange(n_samples),
        size=int(n_samples * 0.22),
        replace=False,
    )
    fault[fault_indices] = 1

    # Apply fault patterns
    voltage[fault_indices] -= np.random.uniform(25, 70, len(fault_indices))
    current[fault_indices] -= np.random.uniform(3, 9, len(fault_indices))
    temperature[fault_indices] += np.random.uniform(8, 22, len(fault_indices))
    inverter_efficiency[fault_indices] -= np.random.uniform(8, 20, len(fault_indices))

    # Avoid impossible values
    current = np.clip(current, 0.5, None)
    inverter_efficiency = np.clip(inverter_efficiency, 55, 100)

    # Recalculate power after introducing faults
    power = voltage * current * (inverter_efficiency / 100)

    data = pd.DataFrame(
        {
            "irradiance": irradiance.round(2),
            "temperature": temperature.round(2),
            "voltage": voltage.round(2),
            "current": current.round(2),
            "power": power.round(2),
            "inverter_efficiency": inverter_efficiency.round(2),
            "fault": fault,
        }
    )

    return data


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    df = generate_solar_fault_data()
    output_path = data_dir / "solar_fault_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset generated successfully: {output_path}")
    print(df.head())
