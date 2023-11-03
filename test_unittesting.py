from utils import tune_hparams, get_digits_dataset, split_train_dev_test
from itertools import product

def test_get_data_split():
    x, y = get_digits_dataset()
    assert len(x) > 0
    assert len(y) > 0

def test_tune_hparams():
    x, y = get_digits_dataset()

    gamma_list = [0.001, 0.002, 0.003] 
    c_range_list = [0.1, 0.2, 0.3]

    dev_size = [0.1, 0.2, 0.3] 
    test_size = [0.1, 0.2, 0.3]

    combined_size_list = product(dev_size, test_size)

    for dev_s, test_s in combined_size_list:
        x, y = get_digits_dataset()
        list_combined = product(gamma_list, c_range_list)
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=test_s, dev_size=dev_s)
        best_hparams, best_model, accuracy_score_train, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_combined)