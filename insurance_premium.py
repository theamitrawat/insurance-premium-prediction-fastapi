import pickle
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "insurance_premium_dataset_200_rows.csv"
MODEL_PATH = BASE_DIR / "trained_model" / "model.pkl"


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

df_feat = df.copy()


# ==========================================================
# Feature Engineering
# ==========================================================

# ---------- BMI ----------
df_feat["height_m"] = df_feat["height"] / 100
df_feat["bmi"] = df_feat["weight"] / (df_feat["height_m"] ** 2)


# ---------- Age Group ----------
def age_group(age):
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    else:
        return "senior"


df_feat["age_group"] = df_feat["age"].apply(age_group)


# ---------- Lifestyle Risk ----------
def lifestyle_risk(row):
    is_smoker = row["smoker"] == "Yes"

    if is_smoker and row["bmi"] > 30:
        return "high"
    elif is_smoker and row["bmi"] > 27:
        return "medium"
    else:
        return "low"


df_feat["lifestyle_risk"] = df_feat.apply(
    lifestyle_risk,
    axis=1
)


# ---------- City Tier ----------

tier_1_cities = [
    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Ahmedabad"
]

tier_2_cities = [
    "Jaipur",
    "Lucknow"
]


def city_tier(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    else:
        return 3


df_feat["city_tier"] = df_feat["city"].apply(city_tier)


# ==========================================================
# Final Features
# ==========================================================

df_feat = df_feat[
    [
        "income_lpa",
        "occupation",
        "bmi",
        "age_group",
        "lifestyle_risk",
        "city_tier",
        "insurance_premium_category",
    ]
]


# ==========================================================
# Split Features and Target
# ==========================================================

X = df_feat[
    [
        "income_lpa",
        "occupation",
        "bmi",
        "age_group",
        "lifestyle_risk",
        "city_tier",
    ]
]

y = df_feat["insurance_premium_category"]


# ==========================================================
# Preprocessing
# ==========================================================

categorical_features = [
    "occupation",
    "age_group",
    "lifestyle_risk",
    "city_tier",
]

numerical_features = [
    "income_lpa",
    "bmi",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "num",
            "passthrough",
            numerical_features,
        ),
    ]
)


# ==========================================================
# Pipeline
# ==========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)


# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1,
)


# ==========================================================
# Train Model
# ==========================================================

pipeline.fit(
    X_train,
    y_train,
)


# ==========================================================
# Evaluate
# ==========================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print("=" * 50)
print(f"Model Accuracy : {accuracy:.2%}")
print("=" * 50)


# ==========================================================
# Save Model
# ==========================================================

MODEL_PATH.parent.mkdir(exist_ok=True)

with MODEL_PATH.open("wb") as f:
    pickle.dump(
        pipeline,
        f,
    )

print(f"\nModel saved successfully at:\n{MODEL_PATH}")
