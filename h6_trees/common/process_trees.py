import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def prepare_data(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    y_train = df_train['fuel_efficiency_mpg'].values
    y_val = df_val['fuel_efficiency_mpg'].values
    df_train_feat = df_train.drop(columns=['fuel_efficiency_mpg'])
    df_val_feat = df_val.drop(columns=['fuel_efficiency_mpg'])
    train_dicts = df_train_feat.to_dict(orient='records')
    val_dicts = df_val_feat.to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    return X_train, X_val, y_train, y_val, dv


def get_decision_tree_split(X_train, y_train, dv):
    dt = DecisionTreeRegressor(max_depth=1, random_state=1)
    dt.fit(X_train, y_train)
    feature_names = dv.get_feature_names_out()
    feature_index = dt.tree_.feature[0]
    return feature_names[feature_index]


def evaluate_random_forest(X_train, y_train, X_val, y_val, n_estimators=10, max_depth=None, random_state=1, n_jobs=-1):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    return root_mean_squared_error(y_val, y_pred)


def tune_n_estimators(X_train, y_train, X_val, y_val, n_estimators_list):
    lv_results = {}
    for n in n_estimators_list:
        rmse = evaluate_random_forest(X_train, y_train, X_val, y_val, n_estimators=n)
        lv_results[n] = round(rmse, 3)
        print(f"n_estimators={n}: RMSE = {rmse:.3f}")
    best_n = min(lv_results, key=lambda k: lv_results[k])
    return best_n


def tune_max_depth(X_train, y_train, X_val, y_val, max_depths, n_estimators_list):
    depth_scores = {}
    for depth in max_depths:
        rmses = [
            evaluate_random_forest(X_train, y_train, X_val, y_val, n_estimators=n, max_depth=depth)
            for n in n_estimators_list
        ]
        mean_rmse = np.mean(rmses)
        depth_scores[depth] = mean_rmse
        print(f"max_depth={depth}: Mean RMSE = {mean_rmse:.4f}")
    return min(depth_scores, key=lambda k: depth_scores[k])


def get_most_important_feature(X_train, y_train, dv, target_features):
    rf_fi = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
    rf_fi.fit(X_train, y_train)
    feature_names = dv.get_feature_names_out()
    importances = rf_fi.feature_importances_
    importance_dict = {}
    for name, importance in zip(feature_names, importances):
        orig_name = name.split('=')[0]
        importance_dict[orig_name] = importance_dict.get(orig_name, 0) + importance

    for feat in target_features:
        print(f"{feat}: {importance_dict.get(feat, 0):.4f}")

    return max(target_features, key=lambda f: importance_dict.get(f, 0))


def tune_xgboost_eta(X_train, y_train, X_val, y_val, dv, eta_list=None):
    if eta_list is None:
        eta_list = [0.3, 0.1]
    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    eta_results = {}
    for eta in eta_list:
        xgb_params = {
            'eta': eta,
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'reg:squarederror',
            'nthread': 8,
            'seed': 1,
            'verbosity': 0,
        }
        model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)
        y_pred_xgb = model.predict(dval)
        val_rmse = root_mean_squared_error(y_val, y_pred_xgb)
        eta_results[eta] = val_rmse
        print(f"eta={eta}: Validation RMSE = {val_rmse:.4f}")

    best_eta = min(eta_results, key=lambda k: eta_results[k])
    return best_eta, eta_results[best_eta]
