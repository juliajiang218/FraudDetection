"""
Disclaimer:
This script is provided for csc373-A assignment 6.

Limitations of Liability:
- Under no circumstances shall the authors, contributors, or affiliated institutions be held liable for any
  direct, indirect, incidental, special, or consequential damages arising out of the use or inability to use
  this script or the data processed by it.

Restrictions on Distribution:
- This script is provided under the terms of its license and may not be copied, modified, or redistributed
  without explicit permission from the copyright holders.
- Any unauthorized use, reproduction, or distribution of this script is strictly prohibited.

Copyright Information:
- © 2025 Julia Jiang. All rights reserved.
- This script is protected by copyright law and any unauthorized use or distribution is strictly prohibited.

By using this script, you acknowledge that you have read, understood, and agree to the terms outlined
above.
"""

import pandas as pd
import numpy as np
# from ydata_profiling import ProfileReport
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# report data
def data_report(filepath, outputpath, target='class'):
      """
      Meta Data: Size, Dimension of a dataframe.
      Calls: analyze_data and produces: a report.
      Takes: filepath, outputpath, target (string)
      """
      df = read_csv(filepath)
      size = df.shape[0]
      dimension = df.shape[1]

      result = analyze_data(df, target)

      with open(outputpath, 'w') as f:
        f.write(f"Size: {size}\n")
        f.write(f"Dimension: {dimension}\n")
        f.write(f"Target Column: '{target}'\n")
        f.write("Sample Data:\n")
        f.write(df.head(2).to_string(index=False))
        f.write('\n\nData Summary:\n')
        f.write(result.to_string(index=False))  # Converts the DataFrame to a nicely formatted string


def analyze_data(df, class_column):
    """
    This function returns a df that identifies:
      1. data type(s) of column and its resolution,
      2. index of duplicate rows, 
      3. number of rows with missing values,
      4. each column's distribution (range, mean, median, outlier of top 10%),
      5. correlation of each column with the class label (if numeric).
    """
    result_df = pd.DataFrame(columns=[
        'column', 'dtype',
        'duplicate_rows','num_nulls', 'nulls_percentage',
        'min','max','range', 'median','mean', 'outlier',
        'correlation_with_class',
        'top2_occuring_values'
    ])
    
    size = df.shape[0]

    # Check if class column is valid
    has_class = class_column in df.columns and np.issubdtype(df[class_column].dtype, np.number)

    for c in df.columns:
        col_dtype = df[c].dtype
        num_duplicates = df.duplicated().sum()
        num_nulls = df[c].isnull().sum()
        num_percent = round(num_nulls / size, 2)

        # Initialize values
        min_val = max_val = range_val = median = mean = outlier_cutoff = correlation = None

        # Numeric-only stats
        if np.issubdtype(col_dtype, np.number):
            min_val = round(df[c].min(), 2)
            max_val = round(df[c].max(), 2)
            range_val = max_val - min_val
            median = round(df[c].median(), 2)
            mean = round(df[c].mean(), 2)
            outlier_cutoff = round(df[c].quantile(0.975), 5)

            # Spearman's correlation with class column
            if has_class and c != class_column:
                # Drop NA for fair comparison
                valid = df[[c, class_column]].dropna()
                if len(valid) > 1:
                    spearman_corr, _ = spearmanr(valid[c], valid[class_column])
                    correlation = round(spearman_corr, 3)

        # top 2 occuring values:
        # the top 10 frequently appearing values in the column
        value_counts = df[c].value_counts()
        top2_values = value_counts.head(2).index.tolist()
        top2_frequencies = value_counts.head(2).tolist()
        top2_values_freq_dict = dict(zip(top2_values, top2_frequencies))

        result_df.loc[len(result_df)] = {
            'column': c,
            'dtype': col_dtype,
            'duplicate_rows': num_duplicates,
            'num_nulls': num_nulls,
            'nulls_percentage': num_percent,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'median': median,
            'mean': mean,
            'outlier': outlier_cutoff,
            'correlation_with_class': correlation,
            'top2_occuring_values':top2_values_freq_dict
        }

    return result_df


def read_csv(path):
   "Read a csv file in the directory path."
   df = pd.read_csv(path)
   return df


def custom_split(
    dataset, 
    target_col='class', 
    test_size=0.2, 
    random_state=42, 
    known_outliers=30, 
    cont_rate=0.02, 
    data_format=0
):
    """
    # function borrowed from:
    #    - @author: Guansong Pang
    #    - Referencee Source: https://github.com/GuansongPang/deviation-network/blob/master/devnet.py
    """
    
    # Perform a stratified split.
    train_df, dev_df = train_test_split(
        dataset, test_size=test_size, random_state=random_state, stratify=dataset[target_col]
    )
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    
    # Separate features and labels from training set.
    X_train = train_df.drop(target_col, axis=1).values
    y_train = train_df[target_col].values
    
    # Remove extra anomalies if they exceed known_outliers.
    outlier_indices_train = np.where(y_train == 1)[0]
    n_outliers_train = len(outlier_indices_train)
    rng = np.random.RandomState(random_state)
    if n_outliers_train > known_outliers:
        num_remove = n_outliers_train - known_outliers
        remove_idx = rng.choice(outlier_indices_train, num_remove, replace=False)
        X_train = np.delete(X_train, remove_idx, axis=0)
        y_train = np.delete(y_train, remove_idx, axis=0)
    
    # Compute the number of noise samples to inject.
    inlier_indices_train = np.where(y_train == 0)[0]
    n_inliers_train = len(inlier_indices_train)
    n_noise = int(n_inliers_train * cont_rate / (1 - cont_rate))
    
    # Obtain all anomalies from the entire dataset (to be used for noise injection).
    original_outliers = dataset[dataset[target_col] == 1].drop(target_col, axis=1).values
    
    # Inject noise samples using the appropriate function.
    if data_format == 0:
        # For dense data; ensure you have inject_noise imported from your devnet module.
        from devnet import inject_noise
        noises = inject_noise(original_outliers, n_noise, random_state)
        X_train = np.append(X_train, noises, axis=0)
        noise_labels = np.zeros(n_noise)
        y_train = np.append(y_train, noise_labels)
    else:
        # For sparse data.
        from scipy.sparse import vstack
        from devnet import inject_noise_sparse
        noises = inject_noise_sparse(original_outliers, n_noise, random_state)
        X_train = vstack([X_train, noises])
        noise_labels = np.zeros((noises.shape[0],))
        y_train = np.append(y_train, noise_labels)
    
    # Convert adjusted training data back into a DataFrame.
    feature_columns = dataset.drop(target_col, axis=1).columns
    train_df_adj = pd.DataFrame(X_train, columns=feature_columns)
    train_df_adj[target_col] = y_train
    
    return train_df_adj, dev_df

def process_data(df, target):
    X_df = df.drop(target, axis=1)
    y = df[target].values
    X = X_df.values
    feature_names = X_df.columns.tolist()
    return X, y, feature_names

def evaluate(true_labels, predicted_scores):

    aucroc = roc_auc_score(true_labels, predicted_scores)
    aucpr =  average_precision_score(true_labels, predicted_scores)
    return aucroc, aucpr

def report_results(name, aucroc, aucpr, f):
    f.write(f"{name}'s AUC-ROC score: %.3f\n" % aucroc)
    f.write(f"{name}'s AUC-PR score: %.3f\n" % aucpr)
    f.write("-" * 40 + "\n")

def features_engineering(X, y, cols_to_drop, feature_names):
    # Convert X and y to DataFrame with proper headers
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.Series(y, name='class')

    # Combine into one DataFrame for deduplication
    df = pd.concat([X_df, y_df], axis=1)
    df = df.drop_duplicates()

    # Drop specified columns
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Split again
    y_clean = df['class'].to_numpy()
    X_clean = df.drop(columns=['class']).to_numpy()
    updated_feature_names = df.drop(columns=['class']).columns.tolist()

    return X_clean, y_clean, updated_feature_names

from imblearn.over_sampling import SMOTE
import pandas as pd

def generate_synthetic_data_smote(df, multiplier=2.0, target_col="class", random_state=42):
    """
    Generate synthetic data using repeated SMOTE to reach multiplier * original size.

    Parameters:
        df (pd.DataFrame): Original dataset.
        multiplier (float): Desired total size = multiplier * len(df).
        target_col (str): Name of the target column.
        random_state (int): Random seed.

    Returns:
        tuple: (X_synth, y_synth)
    """

    X_orig = df.drop(columns=[target_col])
    y_orig = df[target_col]
    total_target_samples = int(len(df) * multiplier)

    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_orig, y_orig)

    current_size = len(X_balanced)
    X_list = [X_balanced]
    y_list = [y_balanced]

    while current_size < total_target_samples:
        X_more, y_more = smote.fit_resample(X_orig, y_orig)
        needed = min(len(X_more), total_target_samples - current_size)
        X_list.append(X_more[:needed])
        y_list.append(y_more[:needed])
        current_size += needed

    X_final = pd.concat(X_list, ignore_index=True)[:total_target_samples]
    y_final = pd.concat([pd.Series(y) for y in y_list], ignore_index=True)[:total_target_samples]

    return X_final, y_final

def report_dataset_results(X_test, aucroc, aucpr, file_handle):
    """
    Write the total number of synthetic test samples to the report file.

    Parameters:
        X_test (pd.DataFrame): Synthetic test data.
        aucroc (float): AUC-ROC score.
        aucpr (float): AUC-PR score.
        file_handle: Open file object to write into.
    """
    file_handle.write(f"Test Set Size: {len(X_test)}\n")
    file_handle.write(f"AUC-ROC: {aucroc:.4f}\n")
    file_handle.write(f"AUC-PR:  {aucpr:.4f}\n")
    file_handle.write("-" * 40 + "\n")


