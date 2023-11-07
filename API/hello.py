from flask import Flask, request
from utils import tune_hparams, predict_and_compare
from sklearn import datasets, metrics, svm

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/hello/<value>")
def hello_world(value):
    return "<p>Hello, World!</p>" + value

@app.route("/sum/<x>/<y>")
def sum(x, y):
    sum = int(x) + int(y)
    return str(sum)

@app.route("/checkModel/<x>/<y>", methods=['POST'])
def checkModel(x, y):
    js = request.get_json()
    x = int(js['x'])
    y = int(js['y'])

    return x + y

@app.route("/predict/<image1>/<image2>", methods=['POST'])
def predict(image1, image2):
    # best_hparams, best_model, accuracy_score_train, best_accuracy = tune_hparams(image1, y_train, X_dev, y_dev, list_combined)
    best_model = svm.SVC

    # Predict
    return predict_and_compare(best_model, image1, image2)

if __name__ == "__main__":
    app.run()