from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
figures_save_path = _THIS_DIR.parent / 'figures'
figures_save_path.mkdir(parents=True, exist_ok=True)

def summarize_dataset(df, title, dim1_comment=None, dim2_comment=None, filename="dataset_summary.png"):
    
    # Creates a summary table for a pandas DataFrame and saves it as an image

    summary_rows = []
    threshold = 4  # Cutoff for categorical vs continuous
    save_path = figures_save_path / filename

    # Title
    summary_rows.append([title])
    
    # Overall Dataset Info
    dim1_label = f"dim1 ({dim1_comment})" if dim1_comment else "dim1"
    summary_rows.append([f"{dim1_label} : {df.shape[1]}"])
    
    dim2_label = f"dim2 ({dim2_comment})" if dim2_comment else "dim2"
    summary_rows.append([f"{dim2_label} : {df.shape[0]}"])
    
    summary_rows.append([f"Duplicate rows : {df.duplicated().sum()}"])
    summary_rows.append([f"Total missing values : {df.isna().sum().sum()}"])
    
    # Global Outlier Detection
    total_outliers = 0
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            series_clean = series.dropna()
            if not series_clean.empty and series_clean.nunique() > threshold:
                q1, q3 = series_clean.quantile([0.25, 0.75])
                iqr = q3 - q1
                col_out = len(series_clean[(series_clean < q1 - 1.5*iqr) | (series_clean > q3 + 1.5*iqr)])
                total_outliers += col_out
    
    summary_rows.append([f"Outliers detected (Method: IQR) : {total_outliers}"])

    summary_rows.append([""]) # Spacer
    
    # Feature Details
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_count = series.nunique()
        
        # Determine Feature Type
        if is_numeric and unique_count > threshold:
            # Column Header Row
            summary_rows.append([f"{col} - Continuous - {series.dtype}"])
            
            # Stats
            summary_rows.append([f"  Mean \u00b1 Std : {series.mean():.2f} \u00b1 {series.std():.2f}"])
            
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            col_outliers = len(series[(series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)])
            summary_rows.append([f"  Outliers (IQR) : {col_outliers}"])
        else:
            type_label = "Binary" if unique_count <= 2 else "Categorical"
            summary_rows.append([f"{col} - {type_label} - {series.dtype}"])

            counts = series.value_counts(dropna=True)
            for val, count in counts.items():
                pct = (count / len(series)) * 100
                summary_rows.append([f"  {val} - {count} - {pct:.1f}%"])
        
        # Column-specific NaN info
        nan_pct = (series.isna().sum() / len(series)) * 100
        summary_rows.append([f"  NaN - {series.isna().sum()} - {nan_pct:.1f}%"])
        summary_rows.append([""]) # Spacer
        
    # --- VISUALIZATION ---
    summary_df = pd.DataFrame(summary_rows)
    fig_height = max(4, len(summary_rows) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')
    
    # Table creation
    table = ax.table(cellText=summary_df.values, loc='center', cellLoc='left', edges='open')
    
    # Global Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)
    
    # Row customization
    for i, row in enumerate(summary_rows):
        # content = str(row[0])

        if i == 0: # Title
            table[(i, 0)].set_text_props(ha='center', weight='bold')
        elif 1 <= i <= 5: # Overall dataset info
            table[(i, 0)].set_text_props(weight='bold')
        elif i > 6 and row[0] != "" and not row[0].startswith("  "): # Bold Column Headers
            table[(i, 0)].set_text_props(weight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_distributions(df, target='num', filename="feature_distributions.png"):
    
    # Generates boxplots to assess distributions and outliers

    save_path = figures_save_path / filename

    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(num_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(df, filename="correlation_heatmap.png"):

    # Performs correlation analysis between features

    save_path = figures_save_path / filename

    plt.figure(figsize=(12, 10))
    correlation = df.select_dtypes(include='number').corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_pca_separability(df, target='num', filename="pca_separability.png"):

    # Dimensionality reduction to sense class separability

    save_path = figures_save_path / filename

    # Preprocessing for PCA: Drop rows with NaNs and scale
    temp_df = df.dropna()
    X = temp_df.drop(columns=[target])
    y = temp_df[target]
    
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    plt.title("PCA: 2D Projection for Class Separability")
    plt.colorbar(label='Heart Disease (num)')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_kernel_pca(df, target='num', filename="kernel_pca.png"):

    # Applies RBF kernel to find non-linear separability

    save_path = figures_save_path / filename

    temp_df = df.dropna()
    X = temp_df.drop(columns=[target])
    y = temp_df[target]
    
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    
    # Using RBF kernel to project into non-linear space
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)
    components = kpca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], c=y, cmap='plasma', edgecolors='k')
    plt.title("Kernel PCA (RBF): Seeking Non-linear Separability")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()