from utils import get_hyperparameter_combinations, split_train_dev_test,get_digits_dataset, tune_hparams, get_preprocess_data
import os

def test_for_hparam_cominations_count():
    # a test case to check that all possible combinations of paramers are indeed generated
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    h_p={}
    h_p['gamma'] = gamma_list
    h_p['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_p)
    
    assert len(h_params_combinations) == len(gamma_list) * len(C_list)

def create_test_hyperparameter():
    gamma_list = [0.001, 0.01]
    C_list = [1]
    h_p={}
    h_p['gamma'] = gamma_list
    h_p['C'] = C_list
    h_p_combinations = get_hyperparameter_combinations(h_p)
    return h_p_combinations

def create_dummy_data():
    X, y = get_digits_dataset()
    
    X_train = X[:100,:,:]
    y_train = y[:100]
    X_dev = X[:50,:,:]
    y_dev = y[:50]

    X_train = get_preprocess_data(X_train)
    X_dev = get_preprocess_data(X_dev)

    return X_train, y_train, X_dev, y_dev

def test_for_hp_cominations_values():    
    h_p_combinations = create_test_hyperparameter()
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in h_p_combinations) and (expected_param_combo_2 in h_p_combinations)

def test_model_saving():
    X_train, y_train, X_dev, y_dev = create_dummy_data()
    h_p_combinations = create_test_hyperparameter()

    best_hparams, best_model_path, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, 
        y_dev, h_p_combinations)   
    
    assert os.path.exists(best_model_path)

def test_data_splitting():
    X, y = get_digits_dataset()
    
    X = X[:100,:,:]
    y = y[:100]
    
    test_size = .1
    dev_size = .6

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)

    assert (len(X_train) == 30) 
    assert (len(X_test) == 10)
    assert ((len(X_dev) == 60))