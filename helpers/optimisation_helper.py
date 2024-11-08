import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


class OptimiserHelper:

    def __init__(self, X, y):
        # self.model = model
        self.X = X
        self.y = y

    def objective_RF_regr(self, trial):
        # 2. Suggest values of the hyperparameters using a trial object.
        n_estimators = trial.suggest_int('n_estimators', 2, 500)
        max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf)
        return np.mean(
            cross_val_score(rf, self.X, self.y, n_jobs=-1, cv=3, scoring='neg_root_mean_squared_error'))

    def objective_RF_clf(self, trial):
        # 2. Suggest values of the hyperparameters using a trial object.
        n_estimators = trial.suggest_int('n_estimators', 2, 500)
        max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)
        return np.mean(
            cross_val_score(rf, self.X, self.y, n_jobs=-1, cv=3, scoring='roc_auc'))

    def objective_XGB_regr(self, trial):
        # 2. Suggest values of the hyperparameters using a trial object.
        n_estimators = trial.suggest_int('n_estimators', 2, 500)
        max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
        eta = trial.suggest_loguniform('eta', 0.001, 1)
        subsample = trial.suggest_loguniform('subsample', 0.001, 1)
        colsample_bytree = trial.suggest_loguniform('colsample_bytree', 0.001, 1)

        xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, eta=eta, subsample=subsample,
                           colsample_bytree=colsample_bytree)
        return np.mean(
            cross_val_score(xgb, self.X, self.y, n_jobs=-1, cv=3, scoring='neg_root_mean_squared_error'))

    def optimise(self, model):
        if model == 'RF_regr':
            objective = self.objective_RF_regr
        if model == 'RF_clf':
            objective = self.objective_RF_clf
        if model == 'XGB_regr':
            objective = self.objective_XGB_regr
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        return study.best_params
