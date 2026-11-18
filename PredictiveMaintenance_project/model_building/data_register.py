import pandas as pd

RAW_PATH = "PredictiveMaintenance_project/data/engine_data.csv"

# Loading the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it

expected_columns = [
        "Engine_RPM", "Lub_Oil_Pressure", "Fuel_Pressure",
        "Coolant_Pressure", "Lub_Oil_Temperature",
        "Coolant_Temperature", "Engine_Condition"
    ]

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Engine Condition Prediction")
print(df["Engine_Condition"].describe())
