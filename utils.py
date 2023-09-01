from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm

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
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=dev_size, random_state=1)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Model training
def train_model(x, y, model_params, model_type='svm'):
    if model_type == 'svm':
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    # Learn the digits on the train subset
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
    confusion_matrix = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    return predicted, model_classification, confusion_matrix