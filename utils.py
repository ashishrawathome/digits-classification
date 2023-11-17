from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm, tree
from joblib import dump

def get_combinations(param_name, param_values, base_combinations):    
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_lists):    
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations

# Preprocess data
def get_preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Dataset split
def get_data_split(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=dev_size/(1-test_size), random_state=1)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def train_model(x, y, model_params, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC

    if model_type == 'tree':
        clf = tree.DecisionTreeClassifier

    model = clf(**model_params)  
    model.fit(x, y)
    return model

def get_digits_dataset():
    digits = datasets.load_digits()
    # flatten the images
    x = digits.images
    y = digits.target
    return x, y

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)

    model_classification = metrics.classification_report(y_test, predicted)
    accuracy_score = metrics.accuracy_score(y_test, predicted)
    precision_score = metrics.precision_score(y_test, predicted, average='macro')

    # Print Confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, predicted)
    print(conf_matrix)

    print("----- Accuracy Score ----- ")
    print(accuracy_score)
    print("----- Confusion Matrix ----- ")
    print(conf_matrix)
    print("----- Precision Score ----- ")
    print(precision_score)

    # F1-score
    f1_score = metrics.f1_score(y_test, predicted, average='macro')
    print("----- F1-Score ----- ")
    print(f1_score)

    return predicted, model_classification, accuracy_score, f1_score, conf_matrix

def tune_hparams(X_train, Y_train, X_dev, y_dev, list_of_all_param_combination, model_type='svm'):
    best_accuracy = -1
    accuracy_score_dev = -1
    best_model = None
    best_model_path = ""
    best_hparams = {-1, -1}

    for params in list_of_all_param_combination:
        cur_m = train_model(X_train, Y_train, params, model_type)
        predicted, model_classification, accuracy_score_dev, f1_score, confusion_matrix = predict_and_eval(cur_m, X_dev, y_dev)

        if accuracy_score_dev > best_accuracy:
            best_accuracy = accuracy_score_dev
            best_model = cur_m
            best_hparams = params
            best_model_path = "./models/{}_".format(str(model_type)) +"_".join(["{}:{}".format(k,v) for k,v in params.items()]) + ".joblib"
    
    dump(best_model, best_model_path) 

    return best_hparams, best_model_path, best_model, best_accuracy