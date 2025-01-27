import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import mlflow
import mlflow.sklearn

# Load data from csv file
data = pd.read_csv("data/processed/titanicp.csv")
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model and hyperparameters
model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

# Start an MLflow run
with mlflow.start_run():
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Save the best model
    joblib.dump(best_model, "src/models/best_model.pkl")

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(best_model, "model")

    # Save metrics to a JSON file
    metrics = {"accuracy": accuracy}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

