import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def car_data_overview(data_frame):
    print(data_frame.info())
    print('.......................................')


def print_nan_counts(data_frame):
    missing_sum = data_frame.isnull().sum()
    print(f"Missing values in each column:\n{missing_sum}")
    print('.....................................')
    for column in data_frame.columns:
        nan_sum = data_frame[column].isna().sum()
        if nan_sum > 0:
            print(column, 'column nan value sum is: ', nan_sum)


def plot_fuel_efficiency(data_frame):
    sns.histplot(data_frame.fuel_efficiency_mpg)
    plt.title('Distribution of Fuel Efficiency (MPG)')
    plt.xlabel('Fuel Efficiency (MPG)')
    plt.ylabel('Count')
    plt.savefig("results/fuel_efficiency_dist.png", dpi=300, bbox_inches="tight")
    # plt.show()

    log_fuel_efficiency = np.log1p(data_frame.fuel_efficiency_mpg)
    sns.histplot(log_fuel_efficiency)
    plt.title('Distribution of Fuel Efficiency (MPG) with np.log1p')
    plt.xlabel('Fuel Efficiency (MPG)')
    plt.ylabel('Count')
    plt.savefig("results/fuel_efficiency_dist_log1p.png", dpi=300, bbox_inches="tight")
    # plt.show()


def select_columns(data_frame):
    df = data_frame.copy()
    new_df = df.loc[:, ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']]
    new_df.columns = new_df.columns.str.lower().str.replace(' ', '_')
    print_nan_counts(new_df)
    return new_df


def median_horsepower_calc(data_frame):
    print(type(data_frame))
    print('.....................................')
    print('1- Median (50% percentile) horsepower using quantile')
    median_horsepower = data_frame['horsepower'].quantile(0.5)
    print(f"Median horsepower:{median_horsepower}")
    print('.....................................')
    print('2- Median (50% percentile) horsepower through describe method ')
    print(data_frame['horsepower'].describe())
    print('.....................................')


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    error = y - y_pred
    mse = (error ** 2).mean()
    return float(np.sqrt(mse))


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]


def train_linear_regression_lecture(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv @ X.T @ y
    return w[0], w[1:]


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T @ X
    reg = r * np.identity(XTX.shape[0])
    # Option to not regularize bias term w0: reg[0, 0] =
    # =r Only feature coefficients ($w_1, w_2, \dots$) are penalized.
    # = 0 Both feature coefficients and the intercept $w_0$ are penalized
    reg[0, 0] = 0
    XTX_inv = np.linalg.inv(XTX + reg)
    w = XTX_inv @ X.T @ y
    return w[0], w[1:]


import numpy as np


def rmse(y, y_pred):
    error = y - y_pred
    mse = (error ** 2).mean()
    return np.sqrt(mse)


def data_prep_split_42(data_frame):
    data_frame.columns = data_frame.columns.str.lower().str.replace(' ', '_')
    car_data_overview(data_frame)

    n = len(data_frame)
    print(f"size of df: {n}")

    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    print(f"Validation size 20%: {n_val}")
    print(f"Test size 20%:       {n_test}")
    print(f"Train size 60%:      {n_train}")
    print('.....................................')

    np.random.seed(42)
    idx = np.arange(n)
    np.random.shuffle(idx)

    # Fixed .copy() missing parentheses on df_val
    df_train = data_frame.iloc[idx[:n_train]].copy()
    df_val = data_frame.iloc[idx[n_train:n_train + n_val]].copy()
    df_test = data_frame.iloc[idx[n_train + n_val:]].copy()

    y_train = df_train['fuel_efficiency_mpg'].values
    y_val = df_val['fuel_efficiency_mpg'].values

    features = [c for c in df_train.columns if c != 'fuel_efficiency_mpg']

    # Impute with 0
    X_tr_0 = df_train[features].copy()
    X_v_0 = df_val[features].copy()
    X_tr_0['horsepower'] = X_tr_0['horsepower'].fillna(0)
    X_v_0['horsepower'] = X_v_0['horsepower'].fillna(0)

    w0, w = train_linear_regression(X_tr_0.values, y_train)
    y_pred_0 = w0 + X_v_0.values @ w
    score_0 = round(rmse(y_val, y_pred_0), 3)

    # Impute with Mean
    X_tr_m = df_train[features].copy()
    X_v_m = df_val[features].copy()
    mean_hp = X_tr_m['horsepower'].mean()
    X_tr_m['horsepower'] = X_tr_m['horsepower'].fillna(mean_hp)
    X_v_m['horsepower'] = X_v_m['horsepower'].fillna(mean_hp)

    w0, w = train_linear_regression(X_tr_m.values, y_train)
    y_pred_m = w0 + X_v_m.values @ w
    score_mean = round(rmse(y_val, y_pred_m), 3)
    print('************************************')
    print('Seed 42 RMSE scores')
    print(f"RMSE (Fill with 0):    {score_0}")
    print(f"RMSE (Fill with Mean): {score_mean}")
    print('************************************')
    X_train = df_train[features].copy()
    X_val = df_val[features].copy()
    X_train = X_train.fillna(0).values
    X_val = X_val.fillna(0).values
    r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
    print('************************************')
    print('Best r (minimum RSE) from [0, 0.01, 0.1, 1, 5, 10, 100]')
    results = {}
    for r in r_values:
        w0, w = train_linear_regression_reg(X_train, y_train, r=r)
        y_pred = w0 + X_val @ w
        score = round(rmse(y_val, y_pred), 4)
        results[r] = score
        print(f"r = {r:<5} | RMSE = {score}")

    best_r = min(results, key=results.get)
    print(f"\nBest r: {best_r} with RMSE = {results[best_r]}")
    print('************************************')


def data_split_seeds_selection(data_frame):
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    scores = []
    df_seeds = data_frame.copy()
    n = len(df_seeds)
    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)
    features = [c for c in data_frame.columns if c != 'fuel_efficiency_mpg']
    for seed in seeds:
        idx = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(idx)
        df_shuffled = data_frame.iloc[idx].reset_index(drop=True)
        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
        y_train = df_train['fuel_efficiency_mpg'].values
        y_val = df_val['fuel_efficiency_mpg'].values
        X_train = df_train[features].fillna(0).values
        X_val = df_val[features].fillna(0).values
        w0, w = train_linear_regression(X_train, y_train)
        y_pred = w0 + X_val @ w
        score = rmse(y_val, y_pred)
        scores.append(score)

    std_score = np.std(scores)
    std_rounded = round(std_score, 3)
    print('************************************')
    print('Seed selection:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] respective RMSE scores and standard deviation')
    print("RMSE scores per seed:", [round(float(s), 3) for s in scores])
    print(f"Standard deviation of scores: {std_rounded}")
    print('************************************')


def data_split_seed_nine(data_frame):
    seed = 9
    df_seed_nine = data_frame.copy()
    n = len(df_seed_nine)
    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_shuffled_nine = data_frame.iloc[idx].reset_index(drop=True)
    df_train = df_shuffled_nine.iloc[:n_train].copy()
    df_val = df_shuffled_nine.iloc[n_train:n_train + n_val].copy()
    df_test = df_shuffled_nine.iloc[n_train + n_val:].copy()
    df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)

    y_full_train = df_full_train['fuel_efficiency_mpg'].values
    y_test = df_test['fuel_efficiency_mpg'].values

    features = [c for c in data_frame.columns if c != 'fuel_efficiency_mpg']
    X_full_train = df_full_train[features].fillna(0).values
    X_test = df_test[features].fillna(0).values

    w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
    y_pred_test = w0 + X_test @ w
    test_rmse = round(rmse(y_test, y_pred_test), 3)
    print('************************************')
    print('Seed nine Test RMSE scores')
    print(f"Test RMSE: {test_rmse}")
    print('************************************')