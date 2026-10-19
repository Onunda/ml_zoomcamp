from contextlib import redirect_stdout
import pandas as pd

from h4_eval_metrics_classification.common.dataprep import data_overview, fill_missing
from h4_eval_metrics_classification.common.evalmetrics import evaluate_numerical_auc, plot_precision_recall_curves, \
    evaluate_threshold_f1, kfold_cross_validation, tune_regularization, train_evaluate_model, split_data_thrice

if __name__ == "__main__":
    data_Lead_score= pd.read_csv("./data/course_lead_scoring_2026.csv")
    df_ls = data_Lead_score

    with open("./results/homework4_output.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("Data overview")
            print("..............................")
            data_overview(df_ls)
            populated_df = fill_missing(df_ls)
            df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = split_data_thrice(populated_df)
            numerical_vars = ["lead_score", "number_of_courses_viewed", "interaction_count", "annual_income"]
            evaluate_numerical_auc(df_train, y_train, numerical_vars)
            features = [col for col in df_train.columns if col != 'converted']
            model, dv, y_pred, auc = train_evaluate_model(df_train, df_val, y_train, y_val, features, C=1.0)
            intersect_thresh = plot_precision_recall_curves(y_val, y_pred)
            evaluate_threshold_f1(y_val, y_pred, candidate_thresholds=[0.21, 0.41, 0.61, 0.81])
            kfold_cross_validation(df_full_train, features, C=1.0)

