# data_prep.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

ARTIFACT_DIR = "artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_names.joblib")
BACKGROUND_PATH = os.path.join(ARTIFACT_DIR, "X_train_bg.joblib")

GMSC_NUMERIC = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]

TARGET = "SeriousDlqin2yrs"

def load_data(path="data/gmsc_credit_data.csv"):
    df = pd.read_csv(path)
    missing = set(GMSC_NUMERIC + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df.loc[df["age"] < 18, "age"] = np.nan
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1.0)
    df["delinq_count"] = (
        df[["NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse"]].fillna(0) > 0
    ).sum(axis=1)
    df["income_per_dep"] = df["MonthlyIncome"] / (df["NumberOfDependents"].replace({0: np.nan}))
    return df

def build_preprocessor(df: pd.DataFrame):
    num_cols = [c for c in df.columns if c != TARGET]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, num_cols)])
    preprocessor.fit(df.drop(columns=[TARGET]))
    feature_names = num_cols
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    return preprocessor, feature_names

def prepare_and_split(path="data/gmsc_credit_data.csv", test_size=0.2, random_state=42):
    df = load_data(path)
    df = basic_clean(df)
    df = feature_engineering(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    preprocessor, feature_names = build_preprocessor(train_df)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    bg = X_train_t[np.random.choice(X_train_t.shape[0], size=min(200, X_train_t.shape[0]), replace=False)]
    joblib.dump(bg, BACKGROUND_PATH)
    return X_train_t, X_test_t, y_train.values, y_test.values, preprocessor, feature_names

if __name__ == "__main__":
    prepare_and_split()
