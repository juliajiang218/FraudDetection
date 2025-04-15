"""
This file builds a pipeline that uses the custom DEVNET model.
It loads the credit card fraud dataset, uses a custom split function from the utils module,
trains the model using custom_devnet, and finally makes predictions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import dump, load
from utils import custom_split
from devnet_class import CustomDevNet

output_dir = "/deac/csc/classes/csc373/jianb21/assignment_6/output"
data_file = "/deac/csc/classes/csc373/jianb21/assignment_6/data/creditcardfraud_normalised.csv"

def load_creditcard_data(file_path):
    """
    Loads the credit card fraud dataset from CSV.
    Assumes that the CSV file is already preprocessed and includes the target column, 
    named 'class' (0 for normal/inlier, non-zero for anomaly/outlier).
    """
    data = pd.read_csv(file_path)
    return data

def main():
    # Create output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset.
    dataset = load_creditcard_data(data_file)
    
    # Remove duplicate rows.
    # dataset = dataset.drop_duplicates().reset_index(drop=True)
    
    # Overall dataset metrics.
    total_samples = dataset.shape[0]
    total_anomalies = dataset[dataset['class'] != 0].shape[0]
    anomaly_pct_total = (total_anomalies / total_samples) * 100
    
    overall_report = (
        "=== Overall Dataset Metrics ===\n"
        f"Total samples (after duplicate removal): {total_samples}\n"
        f"Total anomalies: {total_anomalies} ({anomaly_pct_total:.4f}%)\n"
        "================================\n\n"
    )
    print(overall_report)
    
    # Use the updated custom split function.
    # Adjust parameters: known_outliers=30, cont_rate=0.02.
    train_df, dev_df = custom_split(
        dataset=dataset,
        target_col='class',
        test_size=0.2,
        random_state=42,
        known_outliers=30,
        cont_rate=0.02,
        data_format=0
    )
    
    # Compute split metrics.
    samples_used = train_df.shape[0] + dev_df.shape[0]
    training_samples = train_df.shape[0]
    development_samples = dev_df.shape[0]
    train_anomalies = train_df[train_df['class'] != 0].shape[0]
    anomaly_pct_train = (train_anomalies / training_samples) * 100 if training_samples > 0 else 0
    
    split_report = (
        "=== Split Dataset Metrics ===\n"
        f"Total samples used in experiment: {samples_used}\n"
        f"Training samples: {training_samples}\n"
        f"Development samples: {development_samples}\n"
        f"Percentage of anomalies in the entire dataset: {anomaly_pct_total:.4f}%\n"
        f"Percentage of anomalies in the training dataset: {anomaly_pct_train:.4f}%\n"
        "================================\n\n"
    )
    print(split_report)
    
    # Save the report to a file.
    report_file = os.path.join(output_dir, "experiment_report.txt")
    with open(report_file, "w") as f:
        f.write(overall_report)
        f.write(split_report)
    
    # Extract features and target.
    X_train = train_df.drop('class', axis=1).values
    y_train = train_df['class'].values
    X_dev = dev_df.drop('class', axis=1).values
    y_dev = dev_df['class'].values
    
    # Build the pipeline with the DEVNET model.
    pipeline = Pipeline([
        ('devnet', CustomDevNet(
            architecture='shallow',  # 'shallow' by default (one hidden layer with 20 units)
            batch_size=512,
            nb_batch=20,
            epochs=50,
            random_seed=42,
            data_format=0
        ))
    ])
    
    print("Starting training pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Save the trained pipeline.
    pipeline_file = os.path.join(output_dir, "devnet_pipeline_initial.pkl")
    dump(pipeline, pipeline_file)
    print(f"Saved pipeline to {pipeline_file}\n")
    
    # Load the pipeline for prediction.
    loaded_pipeline = load(pipeline_file)
    print("Pipeline loaded for prediction.")
    
    # Predict on the development set.
    scores = loaded_pipeline.predict(X_dev)
    roc_auc = roc_auc_score(y_dev, scores)
    pr_auc = average_precision_score(y_dev, scores)
    
    outcome_report = (
        "=== Outcome Metrics ===\n"
        f"AUC-ROC: {roc_auc:.4f}\n"
        f"AUC-PR:  {pr_auc:.4f}\n"
        "=======================\n"
    )
    print(outcome_report)
    
    # Append outcome metrics to the report file.
    with open(report_file, "a") as f:
        f.write(outcome_report)

if __name__ == "__main__":
    main()