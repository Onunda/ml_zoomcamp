import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold


def split_data_thrice(df, target_col='converted', test_size=0.2, val_size=0.25, random_state=1):
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, random_state=random_state)
    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values
    return df_full_train, df_train, df_val, df_test, y_train, y_val, y_test


def evaluate_numerical_auc(df_train, y_train, numerical_variables):
    auc_results = {}
    for var in numerical_variables:
        score = roc_auc_score(y_train, df_train[var])
        if score < 0.5:
            score = roc_auc_score(y_train, -df_train[var])
        auc_results[var] = score
    sorted_auc = sorted(auc_results.items(), key=lambda x: x[1], reverse=True)
    print("AUC scores from highest to lowest:")
    for var, score in sorted_auc:
        print(f"- {var}: {score:.3f}")
    return sorted_auc


def train_evaluate_model(df_train, df_val, y_train, y_val, features, C=1.0):
    train_dicts = df_train[features].to_dict(orient='records')
    val_dicts = df_val[features].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print('----------------------------------------------')
    print(f'Model AUC: {auc:.3f}')
    print('----------------------------------------------')
    return model, dv, y_pred, auc


def plot_precision_recall_curves(y_val, y_pred, thresholds=None):
    precisions, recalls = [], []
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    for t in thresholds:
        ypred = (y_pred >= t).astype(int)
        tp = (ypred & y_val).sum()
        fp = (ypred & ~y_val).sum()
        fn = (~ypred & y_val).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="orange")
    plt.xlabel("Thresholds")
    plt.ylabel("Score")
    plt.title("Precision and Recall Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results/precision_recall.png", dpi=300, bbox_inches="tight")

    u_indices = [i for i, (p, r) in enumerate(zip(precisions, recalls)) if p > 0 or r > 0]
    u_thresholds = thresholds[u_indices]
    u_precisions = np.array(precisions)[u_indices]
    u_recalls = np.array(recalls)[u_indices]
    best_index = np.argmin(np.abs(u_precisions - u_recalls))
    intersection_threshold = u_thresholds[best_index]
    print(f"The precision and recall curves intersect at threshold: {intersection_threshold:.2f}")
    print('----------------------------------------------')
    return intersection_threshold


def evaluate_threshold_f1(y_val, y_pred, candidate_thresholds):
    best_f1 = -1
    best_threshold = None
    for t in candidate_thresholds:
        ypred = (y_pred >= t).astype(int)
        tp = (ypred & y_val).sum()
        fp = (ypred & ~y_val).sum()
        fn = (~ypred & y_val).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = ((2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0)
        print(f"Threshold: {t:.2f} | F1 Score: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print("-------------------------------------------")
    print(f"Maximal F1 score: {best_f1:.3f} achieved at threshold: {best_threshold:.2f}")
    print('----------------------------------------------')
    return best_threshold, best_f1


def kfold_cross_validation(df_full_train, features, target_col='converted', n_splits=5, C=1.0, random_state=1):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]
        y_train_fold = df_train_fold[target_col].values
        y_val_fold = df_val_fold[target_col].values
        train_dicts = df_train_fold[features].to_dict(orient='records')
        val_dicts = df_val_fold[features].to_dict(orient='records')
        dv = DictVectorizer(sparse=False)
        X_train_fold = dv.fit_transform(train_dicts)
        X_val_fold = dv.transform(val_dicts)
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_fold)
        scores.append(auc)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    tune_regularization(df_full_train, features, c_values=[0.000001, 0.001, 1.0])
    print('----------------------------------------------')
    print(f'C with best mean ROC AUC:{C:<8}')
    print(f'Mean AUC:{mean_score:.3f} | Std:{std_score:.3f}')
    return mean_score, std_score


def tune_regularization(df_full_train, features, c_values, target_col='converted', n_splits=5, random_state=1):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for C in c_values:
        scores = []
        for train_idx, val_idx in kfold.split(df_full_train):
            df_train_fold = df_full_train.iloc[train_idx]
            df_val_fold = df_full_train.iloc[val_idx]
            y_train_fold = df_train_fold[target_col].values
            y_val_fold = df_val_fold[target_col].values
            train_dicts = df_train_fold[features].to_dict(orient='records')
            val_dicts = df_val_fold[features].to_dict(orient='records')
            dv = DictVectorizer(sparse=False)
            X_train_fold = dv.fit_transform(train_dicts)
            X_val_fold = dv.transform(val_dicts)
            model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict_proba(X_val_fold)[:, 1]
            auc = roc_auc_score(y_val_fold, y_pred_fold)
            scores.append(auc)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f'C={C:<8.6f} | Mean AUC: {mean_score:.3f} ± {std_score:.3f}')
