from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets

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