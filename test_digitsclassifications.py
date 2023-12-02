from utils import get_hyperparameter_combinations, split_train_dev_test,get_digits_dataset, tune_hparams, get_preprocess_data
from api.app import app
import os
import pytest
import json
from joblib import load

def test_for_all_logisticregression_models():
    files = [filename for filename in os.listdir('models/') if filename.startswith("M22AIE201_lr_")]
    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    for m in solvers:
        model = load('models/M22AIE201_lr_' + str(m) + '.joblib')

    assert(True)

def test_for_all_logisticregression_models():
    files = [filename for filename in os.listdir('models/') if filename.startswith("M22AIE201_lr_")]
    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    for m in solvers:
        model = load('models/M22AIE201_lr_' + str(m) + '.joblib')
        params = model.get_params()

        assert(m, params['solver'])

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

def test_post_root():
    suffix = "{\"image\": [0.0, 0.0, 5.0, 13.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 13.0, 15.0, 10.0, 15.0, 5.0, 0.0, 0.0, 3.0, 15.0, 2.0, 0.0, 11.0, 8.0, 0.0, 0.0, 4.0, 12.0, 0.0, 0.0, 8.0, 8.0, 0.0, 0.0, 5.0, 8.0, 0.0, 0.0, 9.0, 8.0, 0.0, 0.0, 4.0, 11.0, 0.0, 1.0, 12.0, 7.0, 0.0, 0.0, 2.0, 14.0, 5.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 6.0, 13.0, 10.0, 0.0, 0.0, 0.0]}"
    
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    
    assert response.status_code == 200  
    assert response.get_json()['y_predicted'] == 0

    suffix = "{\"image\": [0.0, 0.0, 0.0, 0.0, 14.0, 13.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 16.0, 2.0, 0.0, 0.0, 0.0, 0.0, 14.0, 16.0, 12.0, 0.0, 0.0, 0.0, 1.0, 10.0, 16.0, 16.0, 12.0, 0.0, 0.0, 0.0, 3.0, 12.0, 14.0, 16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 16.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 13.0, 16.0, 1.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 1

    suffix = "{\"image\": [0.0, 0.0, 0.0, 4.0, 15.0, 12.0, 0.0, 0.0, 0.0, 0.0, 3.0, 16.0, 15.0, 14.0, 0.0, 0.0, 0.0, 0.0, 8.0, 13.0, 8.0, 16.0, 0.0, 0.0, 0.0, 0.0, 1.0, 6.0, 15.0, 11.0, 0.0, 0.0, 0.0, 1.0, 8.0, 13.0, 15.0, 1.0, 0.0, 0.0, 0.0, 9.0, 16.0, 16.0, 5.0, 0.0, 0.0, 0.0, 0.0, 3.0, 13.0, 16.0, 16.0, 11.0, 5.0, 0.0, 0.0, 0.0, 0.0, 3.0, 11.0, 16.0, 9.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 2

    suffix = "{\"image\": [0.0, 0.0, 7.0, 15.0, 13.0, 1.0, 0.0, 0.0, 0.0, 8.0, 13.0, 6.0, 15.0, 4.0, 0.0, 0.0, 0.0, 2.0, 1.0, 13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 15.0, 11.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 12.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0, 8.0, 0.0, 0.0, 0.0, 8.0, 4.0, 5.0, 14.0, 9.0, 0.0, 0.0, 0.0, 7.0, 13.0, 13.0, 9.0, 0.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 3

    suffix = "{\"image\": [0.0, 0.0, 0.0, 1.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 13.0, 6.0, 2.0, 2.0, 0.0, 0.0, 0.0, 7.0, 15.0, 0.0, 9.0, 8.0, 0.0, 0.0, 5.0, 16.0, 10.0, 0.0, 16.0, 6.0, 0.0, 0.0, 4.0, 15.0, 16.0, 13.0, 16.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 16.0, 4.0, 0.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 4

    suffix = "{\"image\": [0.0, 5.0, 12.0, 13.0, 16.0, 16.0, 2.0, 0.0, 0.0, 11.0, 16.0, 15.0, 8.0, 4.0, 0.0, 0.0, 0.0, 8.0, 14.0, 11.0, 1.0, 0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 16.0, 14.0, 0.0, 0.0, 0.0, 0.0, 1.0, 6.0, 6.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 3.0, 0.0, 0.0, 0.0, 1.0, 5.0, 15.0, 13.0, 0.0, 0.0, 0.0, 0.0, 4.0, 15.0, 16.0, 2.0, 0.0, 0.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 5

    suffix = "{\"image\": [0.0, 0.0, 0.0, 12.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 12.0, 7.0, 2.0, 0.0, 0.0, 0.0, 0.0, 13.0, 16.0, 13.0, 16.0, 3.0, 0.0, 0.0, 0.0, 7.0, 16.0, 11.0, 15.0, 8.0, 0.0, 0.0, 0.0, 1.0, 9.0, 15.0, 11.0, 3.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 6

    suffix = "{\"image\": [0.0, 0.0, 7.0, 8.0, 13.0, 16.0, 15.0, 1.0, 0.0, 0.0, 7.0, 7.0, 4.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 13.0, 1.0, 0.0, 0.0, 4.0, 8.0, 8.0, 15.0, 15.0, 6.0, 0.0, 0.0, 2.0, 11.0, 15.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 15.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 5.0, 0.0, 0.0, 0.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 7

    suffix = "{\"image\": [0.0, 0.0, 9.0, 14.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0, 12.0, 14.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0, 0.0, 15.0, 4.0, 0.0, 0.0, 0.0, 3.0, 16.0, 12.0, 14.0, 2.0, 0.0, 0.0, 0.0, 4.0, 16.0, 16.0, 2.0, 0.0, 0.0, 0.0, 3.0, 16.0, 8.0, 10.0, 13.0, 2.0, 0.0, 0.0, 1.0, 15.0, 1.0, 3.0, 16.0, 8.0, 0.0, 0.0, 0.0, 11.0, 16.0, 15.0, 11.0, 1.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 8

    suffix = "{\"image\": [0.0, 0.0, 12.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 16.0, 16.0, 14.0, 0.0, 0.0, 0.0, 0.0, 13.0, 16.0, 15.0, 10.0, 1.0, 0.0, 0.0, 0.0, 11.0, 16.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 7.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 16.0, 9.0, 0.0, 0.0, 0.0, 5.0, 4.0, 12.0, 16.0, 4.0, 0.0, 0.0, 0.0, 9.0, 16.0, 16.0, 10.0, 0.0, 0.0]}"
    inp = json.loads(suffix)
    response = app.test_client().post("http://127.0.0.1:5000/predict", json=inp)
    assert response.get_json()['y_predicted'] == 9