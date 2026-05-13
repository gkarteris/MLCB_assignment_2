from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, 
                             balanced_accuracy_score, f1_score, 
                             recall_score, precision_score,
                             confusion_matrix, average_precision_score)
from sklearn.base import clone

class HeartDiseasernCV:
    def __init__(self, n_rounds=10, n_outer=5, n_inner=3):
        """
        Repeated Nested CV class for Heart Disease Classification. [cite: 53, 54, 55]
        """
        self.n_rounds = n_rounds
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.results = {}

    def get_preprocessing_pipeline(self, cat_categories=None):
        cat_features = ['cp', 'restecg', 'slope', 'thal']
        num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        bin_features = ['sex', 'fbs', 'exang']  # Binary: no encoding needed, only imputation

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(
                categories=cat_categories if cat_categories is not None else 'auto',
                handle_unknown='ignore',
                sparse_output=False))
        ])
        bin_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
            # No scaling: binary features are already in {0,1}
        ])

        return ColumnTransformer(transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features),
            ('bin', bin_transformer, bin_features)
        ])

    def run_assessment(self, X, y, estimator_name, estimator, param_space_func, perform_fs=False, tune=False):
        
        # Executes Task 3 (Generalization) or Task 4 (Feature Selection)
        
        cat_features = ['cp', 'restecg', 'slope', 'thal']
        cat_categories = [sorted(X[col].dropna().unique().tolist()) for col in cat_features]
        all_metrics = []
        feature_counts = None
        feature_names = None

        for r in range(self.n_rounds):
            # Stratified to maintain class distribution
            skf_outer = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=r)
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf_outer.split(X, y)):
                X_train_out, X_test_out = X.iloc[train_idx], X.iloc[test_idx]
                y_train_out, y_test_out = y.iloc[train_idx], y.iloc[test_idx]

                # Inner Loop: Hyperparameter Tuning
                if tune:
                    best_params = self.inner_optimization(
                        X_train_out,
                        y_train_out,
                        estimator,
                        param_space_func,
                        inner_seed=r * self.n_outer + fold_idx,
                        cat_categories=cat_categories
                    )
                
                # Outer Loop: Model Training
                current_clf = clone(estimator)
                if tune:
                    current_clf.set_params(**best_params)
                
                # Define Pipeline steps 
                steps = [('pre', self.get_preprocessing_pipeline(cat_categories=cat_categories))]                
                if perform_fs: # Task 4: Model-agnostic Feature Selection
                    selector = RFECV(
                        estimator=RandomForestClassifier(random_state=42),
                        step=1,
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    )
                    steps.append(('selector', selector))                
                steps.append(('clf', current_clf))
                pipe = Pipeline(steps)

                # Fit on training data only
                pipe.fit(X_train_out, y_train_out)

                if perform_fs:
                    pre_step = pipe.named_steps['pre']
                    selector_step = pipe.named_steps['selector']
                    
                    # Get feature names after preprocessing
                    transformed_names = pre_step.get_feature_names_out()
                    
                    if feature_counts is None:
                        feature_counts = np.zeros(len(transformed_names))
                        feature_names = transformed_names
                    
                    feature_counts += selector_step.support_.astype(int)
                
                # Evaluation on unseen data
                preds = pipe.predict(X_test_out)
                probs = pipe.predict_proba(X_test_out)[:, 1]

                tn, fp, fn, tp = confusion_matrix(y_test_out, preds).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                all_metrics.append({
                    'estimator': estimator_name,
                    'round': r,
                    'fold': fold_idx,
                    'mcc': matthews_corrcoef(y_test_out, preds),
                    'auc': roc_auc_score(y_test_out, probs),
                    'prauc': average_precision_score(y_test_out, probs),
                    'ba': balanced_accuracy_score(y_test_out, preds),
                    'f1': f1_score(y_test_out, preds),
                    'recall': recall_score(y_test_out, preds),
                    'specificity': specificity,
                    'precision': precision_score(y_test_out, preds, zero_division=0)
                })

        if perform_fs and feature_counts is not None:
            feature_stability = pd.Series(
                feature_counts / (self.n_rounds * self.n_outer),
                index=feature_names
            ).sort_values(ascending=False)
            print(feature_stability)


        return pd.DataFrame(all_metrics)

        
    def inner_optimization(self, X, y, estimator, param_space_func, inner_seed=42, cat_categories=None):

        # Optuna-based inner loop for hyperparameter tuning

        def objective(trial):
            params = param_space_func(trial)
            trial.set_user_attr('full_params', json.dumps(params))
            skf_inner = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=inner_seed)
            
            clf_clone = clone(estimator)
            clf_clone.set_params(**params)
            
            inner_pipe = Pipeline([
                ('pre', self.get_preprocessing_pipeline(cat_categories=cat_categories)),
                ('clf', clf_clone)
            ])
            scores = []
            for t_idx, v_idx in skf_inner.split(X, y):
                inner_pipe.fit(X.iloc[t_idx], y.iloc[t_idx])
                scores.append(f1_score(y.iloc[v_idx], inner_pipe.predict(X.iloc[v_idx])))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=inner_seed))
        study.optimize(objective, n_trials=30)
        return json.loads(study.best_trial.user_attrs['full_params'])


    def train_final_model(self, X, y, winner_clf, best_params, cat_categories=None):
        
        # Final Model for deployment (it isn't used in the notebook, but it's here for completeness and future use)

        project_dir = Path(__file__).resolve().parent.parent
        file_name = "best_model.pkl"
        models_dir = project_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        file_path = models_dir / file_name

        final_pipe = Pipeline([
            ('pre', self.get_preprocessing_pipeline(cat_categories=cat_categories)),
            ('clf', clone(winner_clf).set_params(**best_params))
        ])
        final_pipe.fit(X, y)
        joblib.dump(final_pipe, file_path)
        print(f"Final model saved.")
        return final_pipe