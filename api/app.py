from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "models/svm_gamma:0.001_C:1.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"

@app.route("/checkModel/<x>/<y>", methods=['POST'])
def checkModel(x, y):
    js = request.get_json()
    x = int(js['x'])
    y = int(js['y'])

    return x + y

@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    pred = int(model.predict([image]))
    
    return {"y_predicted":pred, "status_code": 200}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)