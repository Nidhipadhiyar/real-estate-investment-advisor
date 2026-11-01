
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset path
DATA_PATH = "data/merged_houseprice_regression_classification.xlsx"


def load_data():
    """Load the real estate ML dataset."""
    df = pd.read_excel(
        DATA_PATH,
        sheet_name="Regression+Class"
    )
    return df


def prepare_data(df):
    """Prepare features and targets for both ML tasks."""

    feature_columns = [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "Year_Built",
        "Location_Score",
        "School_Score",
        "Current_Price"
    ]

    X = df[feature_columns]
    y_class = df["Price_Category"]
    y_reg = df["Price_After_5_Years"]

    return X, y_class, y_reg


def split_data(X, y_class, y_reg):
    """Create separate train-test splits for classification and regression."""

    X_train_class, X_test_class, y_train_class, y_test_class = (
        train_test_split(
            X,
            y_class,
            test_size=0.2,
            random_state=42,
            stratify=y_class
        )
    )

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
        train_test_split(
            X,
            y_reg,
            test_size=0.2,
            random_state=42
        )
    )

    return (
        X_train_class,
        X_test_class,
        y_train_class,
        y_test_class,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg
    )
