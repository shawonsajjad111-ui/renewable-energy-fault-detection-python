# Renewable Energy Fault Detection using Python and Machine Learning

A GitHub-ready Python portfolio project for **AI-based fault detection in renewable energy systems**.  
This project is designed for an Electrical & Electronic Engineering background and connects directly with practical experience in system monitoring, incident analysis, automation, data analytics, and renewable energy reliability.

## Why this project fits my profile

This project demonstrates skills relevant to:

- AI-based fault detection in renewable energy systems
- Predictive maintenance and system reliability
- Solar PV / renewable energy analytics
- Real-time monitoring and incident diagnosis
- Data analysis using Python, Pandas, NumPy, Matplotlib, and Scikit-learn
- Automation-style reporting for engineering operations

## Project Objective

The goal is to detect whether a renewable energy system is operating normally or experiencing a fault using sensor-style data from a simulated solar PV system.

The model classifies system conditions into:

- `0 = Normal`
- `1 = Fault`

## Dataset

The project generates a synthetic solar PV monitoring dataset with realistic engineering features:

| Feature | Meaning |
|---|---|
| irradiance | Solar irradiance in W/m² |
| temperature | Panel/environment temperature in °C |
| voltage | PV system voltage |
| current | PV system current |
| power | Generated power |
| inverter_efficiency | Inverter performance percentage |
| fault | Target label: 0 normal, 1 fault |

Faults are simulated using abnormal voltage drop, current reduction, overheating, low power, and poor inverter efficiency.

## Project Structure

```text
renewable_energy_fault_detection/
│
├── data/
│   └── solar_fault_data.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── power_vs_irradiance.png
│   ├── model_metrics.txt
│   └── predictions.csv
│
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   └── predict_fault.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/renewable-energy-fault-detection.git
cd renewable-energy-fault-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python src/generate_data.py
```

### 5. Train the machine learning model

```bash
python src/train_model.py
```

### 6. Test fault prediction manually

```bash
python src/predict_fault.py
```

## Model Used

This project uses a **Random Forest Classifier** because it is beginner-friendly, works well with tabular sensor data, and gives feature importance for engineering interpretation.

## Example Output

The model produces:

- Accuracy score
- Classification report
- Confusion matrix
- Feature importance chart
- Power vs irradiance chart
- Prediction results CSV

## Engineering Interpretation

A renewable energy monitoring system can use this type of model to identify abnormal operating patterns. For example:

- Low power during high irradiance may indicate panel degradation or wiring issues.
- Low inverter efficiency may indicate inverter fault.
- Abnormal voltage/current behavior may suggest electrical fault or system mismatch.

## Future Improvements

- Use real solar PV datasets
- Add multiple fault categories instead of binary classification
- Build a Streamlit dashboard
- Add live sensor data simulation
- Integrate email/SMS alert automation
- Deploy the project using Docker or cloud services

## Author

**Sajjad Hossain Shawon**  
Electrical & Electronic Engineer | Renewable Energy & Fault Detection Researcher

