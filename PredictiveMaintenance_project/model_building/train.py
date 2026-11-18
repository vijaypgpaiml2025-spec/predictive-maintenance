
# For data manipulation and pipeline building
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# For model training and tuning
import mlflow
import mlflow.xgboost
import xgboost as xgb

# For model serialization
import joblib
import os

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("engine_predictive_maintenance_experiment")

# Load preprocessed datasets from correct artifacts directory
train_path = "PredictiveMaintenance_project/artifacts/train_processed.csv"
test_path = "PredictiveMaintenance_project/artifacts/test_processed.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Processed files not found. Please make sure prepare_data.py has been run.")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Standardized target column is lowercase snake_case
target_col = 'engine_condition'

Xtrain = train_df.drop(columns=[target_col])
ytrain = train_df[target_col]
Xtest = test_df.drop(columns=[target_col])
ytest = test_df[target_col]

# Define numerical features using snake_case to match clean_and_prepare_data output
numeric_features = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]

# Calculate class weight to handle target class imbalance ('engine_condition')
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define preprocessing steps (StandardScaler for all numeric features)
preprocessor = make_column_transformer((StandardScaler(), [f for f in numeric_features if f in Xtrain.columns]))

# Define base XGBoost classifier
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

# Combine preprocessing and model into a Scikit-Learn Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow Run for tracking
with mlflow.start_run():
  # Execute Grid Search with Cross-Validation
  grid_search = GridSearchCV(
      model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1
  )
  grid_search.fit(Xtrain, ytrain)

  # Log all parameter combinations and their validation scores via nested runs
  results = grid_search.cv_results_
  for i in range(len(results["params"])):
    param_set = results["params"][i]
    mean_score = results["mean_test_score"][i]
    std_score = results["std_test_score"][i]

    with mlflow.start_run(nested=True):
      mlflow.log_params(param_set)
      mlflow.log_metric("mean_test_score", mean_score)
      mlflow.log_metric("std_test_score", std_score)

  # Log the best parameters found
  mlflow.log_params(grid_search.best_params_)

  # Extract the best estimator pipeline
  best_model = grid_search.best_estimator_
  print("Best parameters found:", grid_search.best_params_)

  # Evaluate model performance using decision threshold
  classification_threshold = 0.45

  y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
  y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

  y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
  y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

  train_report = classification_report(ytrain, y_pred_train, output_dict=True)
  test_report = classification_report(ytest, y_pred_test, output_dict=True)

  print("\nTest Set Classification Report:")
  print(classification_report(ytest, y_pred_test))

  # Log core evaluation metrics to MLflow
  mlflow.log_metrics({
      "train_accuracy": train_report["accuracy"],
      "train_precision": train_report["1"]["precision"],
      "train_recall": train_report["1"]["recall"],
      "train_f1-score": train_report["1"]["f1-score"],
      "test_accuracy": test_report["accuracy"],
      "test_precision": test_report["1"]["precision"],
      "test_recall": test_report["1"]["recall"],
      "test_f1-score": test_report["1"]["f1-score"],
  })

  # Create deploy folder and save model
  os.makedirs("PredictiveMaintenance_project/deployment", exist_ok=True)
  model_path = "PredictiveMaintenance_project/deployment/best_engine_failure_prediction_model_v1.joblib"
  joblib.dump(best_model, model_path)
  mlflow.log_artifact(model_path, artifact_path="model")
  print(f"Model saved to {model_path}")
