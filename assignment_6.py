"""
@Authors: Julia Jiang, Fiona Zhang
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
- © 2025 Julia Jiang, Fiona Zhang. All rights reserved.
- This script is protected by copyright law and any unauthorized use or distribution is strictly prohibited.

By using this script, you acknowledge that you have read, understood, and agree to the terms outlined
above.
"""
from utils import *
import devnet
from devnet_class import CustomDevNet
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from devnet_utils import *
import os
import joblib

warnings.filterwarnings("ignore")

def main():
    data_filepath = "/deac/csc/classes/csc373/jianb21/assignment_6/data/creditcardfraud_normalised.csv"
    report_output_path = '/deac/csc/classes/csc373/jianb21/assignment_6/outputs/data_report/data_report.txt'
    result_filepath = '/deac/csc/classes/csc373/jianb21/assignment_6/outputs/evaluation/result.txt'
    data_result_filepath = '/deac/csc/classes/csc373/jianb21/assignment_6/outputs/evaluation/unseen_data_performance.txt'
    pipeline_filepath= '/deac/csc/classes/csc373/jianb21/assignment_6/outputs/pipeline/best_model.pkl'

    # data understanding
    data_report(data_filepath, report_output_path)

    # data processing
    df = read_csv(data_filepath)
    
    # train_df is added noise
    target = 'class'
    train_df, test_df = custom_split(df)
    X_train, y_train, features = process_data(train_df, target)
    X_test, y_test, _ = process_data(test_df, target)

    # """
    # start pipeline training
    names = ['baseline', 'DevNet1', 'DevNet2'] 
    anomaly_detectors = [
       IsolationForest(n_estimators=100, max_samples=256, random_state=42), 
       CustomDevNet(
           architecture='shallow',  # 'shallow' by default (one hidden layer with 20 units)
            batch_size=512,
            nb_batch=20,
            epochs=50,
            random_seed=42,
            data_format=0
       ),
       CustomDevNet(
           architecture='shallow',  # 'shallow' by default (one hidden layer with 20 units)
            batch_size=512,
            nb_batch=20,
            epochs=50,
            random_seed=42,
            data_format=0
       )
     ]
    
    pipeline = Pipeline(
        steps=[("scaler", None), ("detector", None)]
    )
  
    # training and finding the best model
    with open(result_filepath, 'w') as f:
      f.write("Anomaly Detector Model Evaluation Results\n")
      f.write("=" * 40 + "\n")
      for i in range(len(names)):
          pipeline.set_params(detector=anomaly_detectors[i])
          if (anomaly_detectors[i] == 'baseline'): 
            pipeline.fit(X_train)
            predictions = -pipeline.score_samples(X_test)
          else:
             if (names[i] == 'DevNet2'):
                cols = ['V4', 'V10', 'V11', 'V12', 'V14']
                print(X_train)
                X_train, y_train, _ = features_engineering(X_train, y_train, cols, features)
                X_test, y_test, _ = features_engineering(X_test, y_test, cols, features)
             
             pipeline.fit(X_train, y_train)

             if (names[i] == 'DevNet1'):
                # dump DevNet1 as pipeline
                joblib.dump(pipeline, pipeline_filepath)
             predictions = pipeline.predict(X_test)
          # evaluate
          aucroc, aucpr = evaluate(y_test, predictions)
          report_results(names[i], aucroc, aucpr, f)
    # """
    # fit the best model with artificially generated datasets and report results
    best_model = joblib.load(pipeline_filepath)
    
    # Evaluate on multiple test sizes (unseen data)
    test_sizes = [0.1, 0.3, 0.5, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0]

    with open(data_result_filepath, 'w') as f:
      f.write("Best Anomaly Detector Model on Unseen Datasets performance Result: \n")
      f.write("=" * 40 + "\n")
      for size in test_sizes:
          X_test, y_test = generate_synthetic_data_smote(df, size)

          predictions = best_model.predict(X_test)
          aucroc, aucpr = evaluate(y_test, predictions)
          report_dataset_results(X_test, aucroc, aucpr, f)


if __name__ == "__main__":
    main()