from flask import Flask
from flask import request
from joblib import load
from markupsafe import escape

app = Flask(__name__)
model_path = "models/svm_gamma:0.001_C:1.joblib"
tree_model_path = "models/tree_max_depth:100.joblib"
lr_model_path = "models/lr_solver:lbfgs.joblib"

model = load(model_path)
tree_model = load(tree_model_path)
lr_model = load(lr_model_path)


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

@app.route("/predict/<m_t>", methods=['POST'])
def predict_digit_with_model(m_t):
    model_type = escape(m_t)
    image = request.json['image']

    # select default model as 'svm'
    selected_model = model

    if model_type == 'tree':
        selected_model = tree_model
    
    if model_type == 'lr':
        selected_model = lr_model

    pred = int(selected_model.predict([image]))

    return {"y_predicted":pred, "status_code": 200}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)