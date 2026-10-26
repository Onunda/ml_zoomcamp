import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("./data/course_lead_scoring_2026.csv")

target = "converted"
categorical_features = ["lead_source", "industry", "employment_status", "location"]
numerical_features = ["number_of_courses_viewed", "annual_income", "interaction_count", "lead_score"]

medians = {}
for col in numerical_features:
    medians[col] = float(df[col].median())
    df[col] = df[col].fillna(medians[col])

for col in categorical_features:
    df[col] = df[col].fillna("NA").astype(str)

X_dict = df[categorical_features + numerical_features].to_dict(orient="records")
y = df[target].values

pipeline = Pipeline([
    ("dv", DictVectorizer(sparse=False)),
    ("model", LogisticRegression(solver="liblinear", C=1.0))
])

pipeline.fit(X_dict, y)
with open("pipeline.bin", "wb") as f_out:
    pickle.dump(pipeline, f_out)

print("pipeline.bin successfully created!")
