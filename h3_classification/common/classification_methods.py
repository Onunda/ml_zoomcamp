from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score


def split_and_prepare_data(data_frame, target_col='converted'):
    df_full_train, df_test = train_test_split(data_frame, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values
    df_train = df_train.drop(columns=[target_col])
    df_val = df_val.drop(columns=[target_col])
    df_test = df_test.drop(columns=[target_col])
    return df_train, df_val, df_test, y_train, y_val, y_test


def analyze_mutual_information_scores(df_train, y_train, categorical_vars):
    print("----------------------------------------------")
    print("Mutual Information Scores:")
    for col in categorical_vars:
        score = mutual_info_score(df_train[col], y_train)
        print(f"{col}: {round(score, 2)}")
    print("----------------------------------------------")


def train_and_evaluate_model(df_train, y_train, df_val, y_val, C=1.0):
    train_dicts = df_train.to_dict(orient='records')
    val_dicts = df_val.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy, model, dv


def perform_feature_elimination(df_train, y_train, df_val, y_val, original_accuracy):
    features = list(df_train.columns)
    scores_diff = {}
    for c in features:
        subset_features = [f for f in features if f != c]
        acc, _, _ = train_and_evaluate_model(df_train[subset_features], y_train, df_val[subset_features], y_val)
        difference = original_accuracy - acc
        scores_diff[c] = difference
        print(f"Without {c:25} -> Accuracy: {acc:.5f} | Difference: {difference:.5f}")
    return scores_diff


def logistic_regression_regularization_c(df_train, y_train, df_val, y_val, c_values):
    results = {}
    print("\n----------------------------------------------")
    print("Train regularized logistic regression using C values: [0.000001, 0.00001, 0.0001, 0.001]):")
    for c in c_values:
        acc, _, _ = train_and_evaluate_model(df_train, y_train, df_val, y_val, C=c)
        rounded_acc = round(acc, 3)
        results[c] = rounded_acc
        print(f"C = {c:.6f} -> Accuracy: {rounded_acc}")
    best_accuracy = max(results.values())
    tied_cs = [c for c, acc in results.items() if acc == best_accuracy]
    best_c = min(tied_cs)
    print(f"\nBest accuracy: {best_accuracy}")
    print(f"Best C values with this accuracy: {tied_cs}")
    print(f"Selected C (smallest among ties): {best_c}")
    print("----------------------------------------------")
    return best_c


def implement_classification_analysis(data_frame):
    df_train, df_val, df_test, y_train, y_val, y_test = split_and_prepare_data(data_frame)
    categorical_vars = ['lead_source', 'industry', 'employment_status', 'location']
    analyze_mutual_information_scores(df_train, y_train, categorical_vars)
    baseline_acc, _, _ = train_and_evaluate_model(df_train, y_train, df_val, y_val, C=1.0)
    print(f'Logistic regression: Accuracy on the validation dataset: {round(baseline_acc, 2)}')
    print("----------------------------------------------")
    print("Feature Elimination Analysis:")
    scores_diff = perform_feature_elimination(df_train, y_train, df_val, y_val, baseline_acc)
    options = ['lead_source', 'number_of_courses_viewed', 'interaction_count']
    smallest_feature = min(options, key=lambda f: scores_diff[f])
    print(f"\nFeature with the smallest difference: {smallest_feature}")
    print(f"Difference value: {scores_diff[smallest_feature]:.3f}")
    c_values = [0.000001, 0.00001, 0.0001, 0.001]
    logistic_regression_regularization_c(df_train, y_train, df_val, y_val, c_values)
