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
from utils import *
import devnet
from devnet_class import CustomDevNet
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from devnet_utils import *
import os

warnings.filterwarnings("ignore")

def main():
    data_filepath = "/deac/csc/classes/csc373/jianb21/assignment_6/data/creditcardfraud_normalised.csv"
    dataset_name = os.path.basename(data_filepath).replace(".csv", "")
    report_output_path = '/deac/csc/classes/csc373/jianb21/assignment_6/outputs/data_report/data_report.txt'
    result_filepath = '/deac/csc/classes/csc373/jianb21/assignment_6/outputs/evaluation/result.txt'

    # data understanding
    data_report(data_filepath, report_output_path)

    # data processing
    df = read_csv(data_filepath)
    
    # train_df is added noise
    target = 'class'
    train_df, test_df = custom_split(df)
    X_train, y_train = process_data(train_df, target)
    X_test, y_test = process_data(test_df, target)
    
    """
    devnet.main([
    '--data_set', dataset_name,
    '--input_path', '/deac/csc/classes/csc373/jianb21/assignment_6/data/',
    '--data_format', '0',
    '--network_depth', '2',
    '--batch_size', '128',
    '--epochs', '20',
    '--known_outliers', '30',
    '--cont_rate', '0.02',
    '--output', '/deac/csc/classes/csc373/jianb21/assignment_6/outputs/evaluation/devnet_result.csv'
    ])
    """

    # start pipeline training
    names = ['baseline', 'DevNet1']#, 'DevNet2', 'DevNet3']
    anomaly_detectors = [
       IsolationForest(n_estimators=100, max_samples=256, random_state=42), 
       CustomDevNet(
           architecture='shallow',  # 'shallow' by default (one hidden layer with 20 units)
            batch_size=512,
            nb_batch=20,
            epochs=50,
            random_seed=42,
            data_format=0
       )]
    
    pipeline = Pipeline(
        steps=[("scaler", None), ("detector", None)]
    )

    # training and finding the best model
    with open(result_filepath, 'w') as f:
      f.write("Anomaly Detector Model Evaluation Results\n")
      f.write("=" * 40 + "\n")
      for i in range(len(names)):
          pipeline.set_params(detector=anomaly_detectors[i])
          pipeline.fit(X_train)
          if (anomaly_detectors[i] == 'baseline'): 
            predictions = -pipeline.score_samples(X_test)
          else:
             predictions = pipeline.predict(X_test)
          # evaluate
          aucroc, aucpr = evaluate(y_test, predictions)
          report_results(names[i], aucroc, aucpr, f)

if __name__ == "__main__":
    main()