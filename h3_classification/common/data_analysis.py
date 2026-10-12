from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def course_conversion_data_overview(data_frame):
    print(data_frame.info())
    print('.......................................')
    print(f"size of df: {len(data_frame)}")
    print('.......................................')
    data_frame.columns = data_frame.columns.str.lower().str.replace(' ', '_')
    check_missing_values(data_frame)
    print(data_frame.head().T)
    print('.......................................')
    print(data_frame.dtypes)

def check_missing_values(data_frame):
    missing_sum = data_frame.isnull().sum()
    print(f"Missing values in each column:\n{missing_sum}")

def fill_missing_values(data_frame):
    categorical_cols = data_frame.select_dtypes(include=['object', 'string']).columns
    data_frame[categorical_cols] = data_frame[categorical_cols].fillna('NA')
    numerical_cols = data_frame.select_dtypes(include=['float64', 'int64']).columns
    data_frame[numerical_cols] = data_frame[numerical_cols].fillna(0.0)
    print(f"Recheck columns if has any missing values after filling:")
    print('.......................................')
    check_missing_values(data_frame)
    print('.......................................')
    return data_frame

def check_mode(data_frame):
    mode_val = data_frame['industry'].mode().iloc[0]
    print(f"Mode for industry column is: {mode_val}")
    print('.......................................')

def correlation_matrix_course_conversion(data_frame):
    numerical_cols = [
        'annual_income',
        'number_of_courses_viewed',
        'interaction_count',
        'lead_score',
        'converted'
    ]
    corr_matrix = data_frame[numerical_cols].corr()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.expand_frame_repr', False)
    print(corr_matrix)
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    strongest_pairs = corr_matrix.where(mask).unstack().dropna()
    top_correlation = strongest_pairs.abs().sort_values(ascending=False).head(1)
    top_pair_name = top_correlation.index[0]
    actual_value = strongest_pairs[top_pair_name]
    print(f"The two features with the biggest correlation: {top_pair_name[0]} and {top_pair_name[1]}")
    print(f"Correlation value being: {actual_value:.2f}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix for Numerical Features')
    plt.tight_layout()
    plt.savefig("./results/corr_matrix_course_conversion.png", dpi=300, bbox_inches="tight")

def split_data_no_converted_anaylse(data_frame):
    df_full_train, df_test = train_test_split(data_frame, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    y_train = df_train['converted'].values
    y_val = df_val['converted'].values
    y_test = df_test['converted'].values

    df_train = df_train.drop(columns=['converted'])
    df_val = df_val.drop(columns=['converted'])
    df_test = df_test.drop(columns=['converted'])

    categorical_vars = ['lead_source', 'industry', 'employment_status', 'location']
    print("----------------------------------------------")

    for col in categorical_vars:
        score = mutual_info_score(df_train[col], y_train)
        print(f"{col}: {round(score, 2)}")

    train_dicts = df_train.to_dict(orient='records')
    val_dicts = df_val.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("----------------------------------------------")
    print(f'Logistic regression: Accuracy on the validation dataset: {round(accuracy, 2)}')
    print("----------------------------------------------")
    features = list(df_train.columns)
    train_dicts = df_train[features].to_dict(orient='records')
    val_dicts = df_val[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    original_accuracy = accuracy_score(y_val, y_pred)
    print(f"Original Accuracy for all features: {original_accuracy}\n")
    scores_diff = {}

    for c in features:
        subset_features = [f for f in features if f != c]
        sub_train_dicts = df_train[subset_features].to_dict(orient='records')
        sub_val_dicts = df_val[subset_features].to_dict(orient='records')
        dv_sub = DictVectorizer(sparse=False)
        X_train_sub = dv_sub.fit_transform(sub_train_dicts)
        X_val_sub = dv_sub.transform(sub_val_dicts)
        sub_model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
        sub_model.fit(X_train_sub, y_train)

        sub_y_pred = sub_model.predict(X_val_sub)
        acc = accuracy_score(y_val, sub_y_pred)
        difference = original_accuracy - acc
        scores_diff[c] = difference
        print(f"Without {c:25} -> Accuracy: {acc:.5f} | Difference: {difference:.5f}")
    options = ['lead_source', 'number_of_courses_viewed', 'interaction_count']
    smallest_feature = min(options, key=lambda f: scores_diff[f])
    print(f"\nFeature with the smallest difference: {smallest_feature}")
    print(f"Difference value: {scores_diff[smallest_feature]}")
    print(f"Difference value(.3f): {scores_diff[smallest_feature]:.3f}")

    features = list(df_train.columns)
    train_dicts = df_train[features].to_dict(orient='records')
    val_dicts = df_val[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    C_values = [0.000001, 0.00001, 0.0001, 0.001]
    results = {}

    for c in C_values:
        model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        rounded_acc = round(acc, 3)
        results[c] = rounded_acc
        print(f"C = {c} -> Accuracy: {rounded_acc}")

    max(results, key=lambda x: (results[x], -x))
    best_accuracy = max(results.values())
    tied_cs = [c for c, acc in results.items() if acc == best_accuracy]
    print(f"\nBest accuracy: {best_accuracy}")
    print(f"Best C values([0.000001, 0.00001, 0.0001, 0.001]) with this accuracy is : {tied_cs}")
    print(f"Selected C (smallest among ties): {min(tied_cs)}")
