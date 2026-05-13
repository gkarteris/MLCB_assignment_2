import pandas as pd
from pathlib import Path
import numpy as np
import sys

from sklearn.model_selection import train_test_split, RandomizedSearchCV

def load_data(file_name):
    project_dir = Path(__file__).resolve().parent.parent
    file_path = project_dir / 'data' / file_name
    if not file_path.is_file():
        print('Reads input file not found. Exiting the program.')
        sys.exit()

    df = pd.read_csv(file_path)
    return df


def split_data_stratified_by_age(df, test_size=0.2, bins=4, seed=42):
    temp_df = df.copy()

    # Identify the age column
    all_age_columns = [col for col in temp_df.columns if 'age' in col.lower()]
    age_columns_count = len(all_age_columns)

    if age_columns_count == 0:
        print("No age columns found. Exiting the program.")
        custom_age_column = input("Specify the exact column name or type 'exit' to quit: ")
        if custom_age_column.lower() == 'exit':
            sys.exit()
        elif custom_age_column in temp_df.columns:
            age_column = custom_age_column
        else:
            print("Invalid column name. Exiting the program.")
            sys.exit()
    else:
        age_column = all_age_columns[0]

    # Stratification
    stratified_age_column = f"{age_column}_bins"
    temp_df[stratified_age_column] = pd.cut(temp_df[age_column], bins=bins, labels=False)

    # Perform the split
    train_set, test_set = train_test_split(
        temp_df, 
        test_size=test_size, 
        stratify=temp_df[stratified_age_column], 
        random_state=seed
    )
    
    # Remove the temporary bin column
    train_set = train_set.drop(columns=[stratified_age_column])
    test_set = test_set.drop(columns=[stratified_age_column])

    return train_set, test_set


def check_zero_variance_features(train_set, val_set, eval_set):
    cpg_cols = [col for col in train_set.columns if col.startswith('cg')]
    
    zero_var_cols = [col for col in cpg_cols if train_set[col].nunique() <= 1]
    
    return zero_var_cols


def create_preprocessing_pipeline(cpg_columns, metadata_columns):
    # Pipeline implemetation for CpG features (numerical)
    cpg_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline implementation for metadata features (categorical)
    metadata_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cpg_transformer', cpg_pipeline, cpg_columns),
            ('categorical_transformer', metadata_pipeline, metadata_columns)
        ]
    )
    return preprocessor

def apply_preprocessing_pipeline(train_set, val_set, eval_set, target_column='age', save_processed_data=False):
    project_dir = Path(__file__).resolve().parent.parent
    metadata_columns = ['sex', 'ethnicity']
    cpg_columns = [col for col in train_set.columns if col.startswith('cg')]

    preprocessor = create_preprocessing_pipeline(cpg_columns, metadata_columns)

    # Start preprocessing by separating features and target variable for all datasets
    X_train, y_train = train_set.drop(columns=[target_column]), train_set[target_column]
    X_val, y_val = val_set.drop(columns=[target_column]), val_set[target_column]
    X_eval, y_eval = eval_set.drop(columns=[target_column]), eval_set[target_column]

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # Transform all datasets
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_eval_proc = preprocessor.transform(X_eval)

    # Get feature names for the transformed datasets
    metadata_columns_names = preprocessor.named_transformers_['categorical_transformer'].named_steps['onehot'].get_feature_names_out(metadata_columns).tolist()
    all_feature_names = cpg_columns + metadata_columns_names

    # Create dataframes for the processed datasets with appropriate column names
    X_train_final = pd.DataFrame(X_train_proc, columns=all_feature_names, index=X_train.index)
    X_val_final = pd.DataFrame(X_val_proc, columns=all_feature_names, index=X_val.index)
    X_eval_final = pd.DataFrame(X_eval_proc, columns=all_feature_names, index=X_eval.index)

    # Split data into three sets based on feature types for model training and evaluation of Task 2
    processed_data_dict = {
        'meta': (X_train_final[metadata_columns_names], X_val_final[metadata_columns_names], X_eval_final[metadata_columns_names]),
        'cpg':  (X_train_final[cpg_columns], X_val_final[cpg_columns], X_eval_final[cpg_columns]),
        'full': (X_train_final, X_val_final, X_eval_final),
        'y':    (y_train, y_val, y_eval)
    }

    # Optional: Save the processed datasets to CSV files
    if save_processed_data:
        save_path = project_dir / 'data'

        if save_path.exists():
            for name, (train, val, eval) in processed_data_dict.items():
                if name != 'y':
                    train.to_csv(save_path / f'X_train_{name}.csv', index=True)
                    val.to_csv(save_path / f'X_val_{name}.csv', index=True)
                    eval.to_csv(save_path / f'X_eval_{name}.csv', index=True)
            y_train.to_csv(save_path / 'y_train.csv', index=True)
            y_val.to_csv(save_path / 'y_val.csv', index=True)
            y_eval.to_csv(save_path / 'y_eval.csv', index=True)

            print("Data saved in CSV files")
        else:
            print(f"Data directory does not exist. Processed data not saved.")

    return processed_data_dict


def save_best_model(model_wrapper, folder_name='models', filename='best_model.pkl'):
    project_dir = Path(__file__).resolve().parent.parent
    save_path = project_dir / folder_name / filename
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_wrapper, f)
        
    print(f"Model successfully saved")