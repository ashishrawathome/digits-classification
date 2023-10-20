"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

###############################################################################
# Get parameter values from command line
# --------------

import matplotlib.pyplot as plt
import pandas as pd
import sys

from utils import get_data_split, get_preprocess_data, train_model, get_digits_dataset, split_train_dev_test, predict_and_eval, tune_hparams, tune_hparams_decision_tree
from sklearn import metrics
from itertools import product

###############################################################################
# Defaults
# --------------

dev_size = [0.1, 0.2, 0.3] 
test_size = [0.1, 0.2, 0.3]

gamma_list = [0.001, 0.002, 0.003] 
c_range_list = [0.1, 0.2, 0.3]

tree_depth = [10, 20, 30, 40, 50] # Decision tree depth

results_dict = {}

###############################################################################
# Get parameter values from command line
# --------------

# python digits.py 

dev_size = float(sys.argv[1])  # First command line argument
test_size = float(sys.argv[2])
gamma_list = float(sys.argv[3])
c_range_list = float(sys.argv[4])
tree_depth = int(sys.argv[5])

###############################################################################
# Preprocess
# --------------

combined_size_list = product([dev_size], [test_size])
list_combined = product([gamma_list], [c_range_list])
counter = 5

# check for different train, dev and test samples
for dev_s, test_s in combined_size_list:
    x, y = get_digits_dataset()

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=test_s, dev_size=dev_s)

    X_train = get_preprocess_data(X_train)
    X_dev = get_preprocess_data(X_dev)
    X_test = get_preprocess_data(X_test)
    
    # Hyper parameter tuning
    best_hparams, best_model, accuracy_score_train, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_combined)

    # Predict
    predicted, model_classification, accuracy_score, confusion_matrix = predict_and_eval(best_model, X_test, y_test)

    results_dict['svm'] = {'model_type': 'svm', 'test_size': test_s, 'dev_size': dev_s, 'train_size': 1 - (test_s + dev_s)
                        , 'train_acc': accuracy_score_train, 'dev_acc': best_accuracy, 'test_acc': accuracy_score}
    
    print("SVM Model => test_size=" + str(test_s) + " dev_size=" + str(dev_s) + " train_size=" + str(1 - (test_s + dev_s)) 
          + " train_acc=" + str(accuracy_score_train) + " dev_acc=" + str(best_accuracy) + " test_acc=" + str(accuracy_score))

    best_hparams, best_model, accuracy_score_train, best_accuracy = tune_hparams_decision_tree(X_train, y_train, X_dev, y_dev, [tree_depth])
    predicted, model_classification, accuracy_score, confusion_matrix = predict_and_eval(best_model, X_test, y_test)

    results_dict['tree'] = {'model_type': 'tree', 'test_size': test_s, 'dev_size': dev_s, 'train_size': 1 - (test_s + dev_s)
                        , 'train_acc': accuracy_score_train, 'dev_acc': best_accuracy, 'test_acc': accuracy_score}

    print("DECISION TREE Model => test_size=" + str(test_s) + " dev_size=" + str(dev_s) + " train_size=" + str(1 - (test_s + dev_s)) 
          + " train_acc=" + str(accuracy_score_train) + " dev_acc=" + str(best_accuracy) + " test_acc=" + str(accuracy_score))
    
df = pd.DataFrame(results_dict)

print(df.describe)