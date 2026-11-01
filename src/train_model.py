
import os
import joblib
import mlflow
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import load_data, prepare_data, split_data


# --------------------------------------------------
# DIRECTORY AND MLFLOW CONFIGURATION
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Real_Estate_Investment_Advisor")


# --------------------------------------------------
# CLASSIFICATION MODEL TRAINING
# --------------------------------------------------

def train_classification_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        print(f"\nTraining Classification Model: {name}")

        with mlflow.start_run(run_name=f"Classification - {name}"):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            macro_f1 = f1_score(
                y_test,
                predictions,
                average="macro"
            )

            # Log experiment parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("task", "classification")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            # Log evaluation metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("macro_f1", macro_f1)

            results[name] = {
                "model": model,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            }

            print("Accuracy:", round(accuracy, 4))
            print("Macro F1:", round(macro_f1, 4))

            print("\nClassification Report:")
            print(classification_report(y_test, predictions))

            print("Confusion Matrix:")
            print(confusion_matrix(y_test, predictions))

    # Select the best classification model
    best_name = max(
        results,
        key=lambda name: results[name]["accuracy"]
    )

    best_model = results[best_name]["model"]

    joblib.dump(
        best_model,
        "models/classification_model.pkl"
    )

    print(f"\nBest Classification Model: {best_name}")
    print("Saved to models/classification_model.pkl")

    return results


# --------------------------------------------------
# REGRESSION MODEL TRAINING
# --------------------------------------------------

def train_regression_models(X_train, X_test, y_train, y_test):

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        print(f"\nTraining Regression Model: {name}")

        with mlflow.start_run(run_name=f"Regression - {name}"):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)

            rmse = mean_squared_error(
                y_test,
                predictions
            ) ** 0.5

            r2 = r2_score(y_test, predictions)

            # Log experiment parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("task", "regression")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            # Log evaluation metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

            results[name] = {
                "model": model,
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            }

            print("MAE:", round(mae, 2))
            print("RMSE:", round(rmse, 2))
            print("R2 Score:", round(r2, 4))

    # Select the best regression model
    best_name = min(
        results,
        key=lambda name: results[name]["mae"]
    )

    best_model = results[best_name]["model"]

    joblib.dump(
        best_model,
        "models/regression_model.pkl"
    )

    print(f"\nBest Regression Model: {best_name}")
    print("Saved to models/regression_model.pkl")

    return results


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Loading dataset...")

    df = load_data()

    print("Dataset shape:", df.shape)

    X, y_class, y_reg = prepare_data(df)

    (
        X_train_class,
        X_test_class,
        y_train_class,
        y_test_class,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg
    ) = split_data(X, y_class, y_reg)

    print("\nTraining samples:", len(X_train_class))
    print("Testing samples:", len(X_test_class))

    # Classification
    print("\n========== CLASSIFICATION ==========")

    classification_results = train_classification_models(
        X_train_class,
        X_test_class,
        y_train_class,
        y_test_class
    )

    # Regression
    print("\n========== REGRESSION ==========")

    regression_results = train_regression_models(
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg
    )

    print("\n========== TRAINING SUMMARY ==========")

    print("\nBest Classification Model:",
          max(
              classification_results,
              key=lambda name: classification_results[name]["accuracy"]
          ))

    print("Best Regression Model:",
          min(
              regression_results,
              key=lambda name: regression_results[name]["mae"]
          ))

    print("\nModel training completed successfully!")
    print("MLflow experiment tracking completed!")
    print("Models saved in the models/ directory.")


if __name__ == "__main__":
    main()
