import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def clean_and_prepare_data(file_path):
    """
    Perform data cleaning, feature engineering, scaling,
    and train-test splitting for the engine predictive maintenance dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check your path.")

    # 1. Load dataset
    df = pd.read_csv(file_path)

    # 2. Data Cleaning: Standardize column names to match snake_case style
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    # 'lub oil temp' becomes 'lub_oil_temp', and 'coolant temp' becomes 'coolant_temp'

    # Remove duplicate rows if any
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

    # Handle missing values (if any present) using median imputation
    if df.isnull().sum().sum() > 0:
        df.fillna(df.median(numeric_only=True), inplace=True)
        print("Missing values imputed with median.")
    else:
        print("No missing values detected.")

    # 3. Feature Engineering
    df['pressure_ratio'] = df['lub_oil_pressure'] / (df['fuel_pressure'] + 1e-5)
    df['temp_differential'] = df['coolant_temp'] - df['lub_oil_temp']

    # 4. Separate Features (X) and Target (y)
    X = df.drop(columns=['engine_condition'])
    y = df['engine_condition']

    # 5. Train-Test Split (Stratified to maintain class proportion)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames for convenience
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 7. Save preprocessed data and scalers as workflow artifacts
    os.makedirs("PredictiveMaintenance_project/artifacts", exist_ok=True)
    joblib.dump(scaler, "PredictiveMaintenance_project/artifacts/scaler.pkl")

    X_train_scaled['engine_condition'] = y_train.values
    X_test_scaled['engine_condition'] = y_test.values

    X_train_scaled.to_csv("PredictiveMaintenance_project/artifacts/train_processed.csv", index=False)
    X_test_scaled.to_csv("PredictiveMaintenance_project/artifacts/test_processed.csv", index=False)

    print("\nData Preparation Complete!")
    print(f"Training Set Shape: {X_train_scaled.shape}")
    print(f"Testing Set Shape: {X_test_scaled.shape}")
    print("Artifacts saved to 'PredictiveMaintenance_project/artifacts/' folder successfully.")

    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    clean_and_prepare_data("PredictiveMaintenance_project/data/engine_data.csv")
