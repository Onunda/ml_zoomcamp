from contextlib import redirect_stdout
import pandas as pd

from h6_trees.common.prep_data import data_brief, fill_missing_zero
from h6_trees.common.process_trees import prepare_data, get_decision_tree_split, evaluate_random_forest, \
    tune_n_estimators, tune_max_depth, get_most_important_feature, tune_xgboost_eta

if __name__ == "__main__":
    data_car_fuel_eff = pd.read_csv("./data/car_fuel_efficiency_2026.csv")
    df_fuel = data_car_fuel_eff

    with open("./results/homework6_output.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("Data overview- Car fuel efficiency data")
            print("..............................")
            data_brief(df_fuel)
            df_filled = fill_missing_zero(df_fuel)
            X_train, X_val, y_train, y_val, dv = prepare_data(df_filled)
            split_feat = get_decision_tree_split(X_train, y_train, dv)
            print(f"1. One-hot or original feature used for splitting: {split_feat}\n")
            rmse_val = evaluate_random_forest(X_train, y_train, X_val, y_val, n_estimators=10)
            print(f"2. RMSE on Validation: {rmse_val:.4f}\n")
            best_n = tune_n_estimators(X_train, y_train, X_val, y_val, [10, 50, 100, 150])
            print(f"3. Lowest RMSE at n_estimators = {best_n}\n")
            best_depth = tune_max_depth(X_train, y_train, X_val, y_val, [10, 15, 20, 25], [10, 50, 100, 150])
            print(f"4. Best max_depth by mean RMSE = {best_depth}\n")
            targets = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
            best_feat = get_most_important_feature(X_train, y_train, dv, targets)
            print(f"5. Most important feature: {best_feat}\n")
            best_eta, best_rmse = tune_xgboost_eta(X_train, y_train, X_val, y_val, dv, [0.3, 0.1])
            print(f"6. Best eta with lowest RMSE: {best_eta} (RMSE = {best_rmse:.4f})")
