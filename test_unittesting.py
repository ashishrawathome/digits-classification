from utils import tune_hparams, get_digits_dataset, split_train_dev_test, get_preprocess_data
from itertools import product

def test_get_data_split():
    x, y = get_digits_dataset()
    assert len(x) > 0
    assert len(y) > 0

def test_tune_hparams():
    dev_size = 0.3
    test_size = 0.3

    gamma_list = 0.002
    c_range_list = 0.2

    combined_size_list = product([dev_size], [test_size])
    list_combined = product([gamma_list], [c_range_list])

    for dev_s, test_s in combined_size_list:
        x, y = get_digits_dataset()
        
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=test_s, dev_size=dev_s)
        X_train = get_preprocess_data(X_train)
        X_dev = get_preprocess_data(X_dev)
        X_test = get_preprocess_data(X_test)
        best_hparams, best_model, accuracy_score_train, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_combined)