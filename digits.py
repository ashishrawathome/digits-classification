"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# Import datasets, classifiers and performance metrics
from utils import get_data_split, get_preprocess_data, train_model, get_digits_dataset, split_train_dev_test, predict_and_eval, tune_hparams, get_combinations, get_hyperparameter_combinations
from sklearn import metrics
from joblib import load

num_runs  = 1
# 1. Get the dataset
X, y = get_digits_dataset()

# 2. Hyperparameter combinations
classifier_param_dict = {}

# SVM
h_params_svm={}
h_params_svm['gamma'] = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
h_params_svm['C'] = [0.1, 1, 10, 100, 1000]
classifier_param_dict['svm'] = get_hyperparameter_combinations(h_params_svm)

# Decision Tree
h_params_tree = {}
h_params_tree['max_depth'] = [5, 10, 15, 20, 50, 100]
classifier_param_dict['tree'] = get_hyperparameter_combinations(h_params_tree)

# Logistic Regression
h_params_lr = {}
h_params_lr['solver'] = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
classifier_param_dict['lr'] = get_hyperparameter_combinations(h_params_lr)

results = []
test_sizes =  [0.2]
dev_sizes  =  [0.2]

for cur_run_i in range(num_runs):
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size              
            X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)
            
            X_train = get_preprocess_data(X_train)
            X_dev = get_preprocess_data(X_dev)
            X_test = get_preprocess_data(X_test)
            
            binary_preds = {}
            model_preds = {}

            for model_type in classifier_param_dict:
                current_hparams = classifier_param_dict[model_type]
                best_hparams, best_model_path, best_m, best_accuracy  = tune_hparams(X_train, y_train, X_dev, y_dev, current_hparams, model_type) 
       
                best_model = load(best_model_path) 

                predicted, model_classification, accuracy_score_train, f1_score_train, conf_matrix = predict_and_eval(best_model, X_train, y_train)
                predicted_test, model_classification_test, accuracy_score_test, f1_score_test, conf_matrix_test = predict_and_eval(best_model, X_test, y_test)
                
                dev_acc = best_accuracy

                print("{}\test_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}, test_f1={:.2f}".format(model_type, test_size, dev_size, train_size, accuracy_score_train, dev_acc, accuracy_score_test, f1_score_test))
                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_acc' : accuracy_score_train, 'dev_acc': dev_acc, 'test_acc': accuracy_score_test}
                results.append(cur_run_results)
                binary_preds[model_type] = y_test == predicted_test
                model_preds[model_type] = predicted_test
                
                print("{}-GroundTruth Confusion metrics".format(model_type))
                print(metrics.confusion_matrix(y_test, predicted_test))


print("svm-tree Confusion metrics".format())
print(metrics.confusion_matrix(model_preds['svm'], model_preds['tree']))

print("binarized predictions")
print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False]))
print("binarized predictions -- normalized over true labels")
print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False] , normalize='true'))
print("binarized predictions -- normalized over pred  labels")
print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False] , normalize='pred'))
