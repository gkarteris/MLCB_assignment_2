import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import bootstrap

rand_seed = 42

def load_data(file_name):
    project_dir = Path(__file__).resolve().parent.parent
    file_path = project_dir / 'data' / file_name
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)

def lr_space(trial):
    # Logistic Regression
    return {
        'C': trial.suggest_float('C', 1e-4, 10, log=True),
        'penalty': 'elasticnet',
        'solver': 'saga',
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'max_iter': 2000,   # saga converges more slowly; raise limit
        'random_state': rand_seed
    }

def rf_space(trial):
    # Random Forest
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': rand_seed
    }

def gnb_space(trial):
    # Gaussian Naive Bayes
    return {
        'var_smoothing': trial.suggest_float('var_smoothing', 1e-11, 1e-6, log=True)
    }

def lda_space(trial):
    # Linear Discriminant Analysis
    shrinkage = trial.suggest_categorical('shrinkage', ['auto', None])
    solver = 'lsqr' if shrinkage == 'auto' else 'svd'
    return {
        'solver': solver,
        'shrinkage': shrinkage
    }

def lgbm_space(trial):
    # LightGBM
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': rand_seed,
        'verbosity': -1
    }

def xgb_space(trial):
    # XGBoost
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': rand_seed,
        'eval_metric': 'logloss',
    }

def catboost_space(trial):
    # CatBoost
    return {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'random_seed': rand_seed,
        'verbose': 0
    }

def median_ci(series, confidence=0.95, n_resamples=9_999, seed=42):
    # Bootstrap percentile CI for the median
    res = bootstrap(
        (series.values,),
        statistic=np.median,
        n_resamples=n_resamples,
        confidence_level=confidence,
        random_state=seed,
        method="percentile",
    )
    return res.confidence_interval.low, res.confidence_interval.high