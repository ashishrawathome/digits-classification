"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from utils import get_data_split, get_preprocess_data, train_model, get_digits_dataset, split_train_dev_test, predict_and_eval, tune_hparams
from sklearn import metrics

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

#_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#for ax, image, label in zip(axes, digits.images, digits.target):
#    ax.set_axis_off()
#    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#    ax.set_title("Training: %i" % label)
gamma_list = [0.001, 0.002, 0.003] 
c_range_list = [0.1, 0.2, 0.3]

from itertools import product

dev_size = [0.1, 0.2, 0.3] 
test_size = [0.1, 0.2, 0.3]

combined_size_list = product(dev_size, test_size)

for dev_s, test_s in combined_size_list:
    x, y = get_digits_dataset()

    list_combined = product(gamma_list, c_range_list)

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=test_s, dev_size=dev_s)

    X_train = get_preprocess_data(X_train)
    X_dev = get_preprocess_data(X_dev)
    X_test = get_preprocess_data(X_test)
    
    best_hparams, best_model, accuracy_score_train, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_combined)
    predicted, model_classification, accuracy_score, confusion_matrix = predict_and_eval(best_model, X_test, y_test)

    print("test_size=" + str(test_s) + " dev_size=" + str(dev_s) + " train_size=" + str(1 - (test_s + dev_s)) 
          + " train_acc=" + str(accuracy_score_train) + " dev_acc=" + str(best_accuracy) + " test_acc=" + str(accuracy_score))


#model = train_model(X_train, y_train, {
# 'gamma': 0.001}, model_type='svm')

#validate 
#predicted = model.predict(X_dev)
#score = model.score(X_dev, y_dev)

#print(
#    f"Model performance on development set {score}\n"
#)

# Predict the value of the digit on the test subset


###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#    ax.set_axis_off()
#    image = image.reshape(8, 8)
#    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

#print(
#    f"Classification report for classifier {best_model}:\n"
#    f"{model_classification}\n"
#)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

#disp = confusion_matrix
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

#plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
#y_true = []
#y_pred = []
#cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
#for gt in range(len(cm)):
#    for pred in range(len(cm)):
#        y_true += [gt] * cm[gt][pred]
#        y_pred += [pred] * cm[gt][pred]

#print(
#    "Classification report rebuilt from confusion matrix:\n"
#    f"{metrics.classification_report(y_true, y_pred)}\n"
#)