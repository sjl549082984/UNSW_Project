import numpy as np
import cv2 as cv

import sklearn.svm as svm
import sklearn.metrics as sm
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score
from sklearn import tree
import pandas as pd
import os
import time

"""
Plant_Phenotyping_Datasets should be placed in the same root directory with this program
"""

# Store file name in dictionary
def search_files(directory1,directory2,label_list):
    object = {}
    for i in label_list:
        object[i] = []
    for curdir, subdirs, files in os.walk(directory1):
        for file in files:
            if file.endswith("rgb.png"):
                label = label_list[0]
                url = os.path.join(curdir, file)
                object[label].append(url)
    for curdir, subdirs, files in os.walk(directory2):
        for file in files:
            if file.endswith("rgb.png"):
                label = label_list[1]
                url = os.path.join(curdir, file)
                object[label].append(url)
    return object

# Normalize the value of the picture
def normalize(input_img):
    minmax_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype='uint8')
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            minmax_img[i, j] = 255 * (input_img[i, j] - np.min(input_img)) / (np.max(input_img) - np.min(input_img))
    return minmax_img

# Preprocess the picture, reduce the noise of the picture
def preprocess(input_img):

    median = cv.medianBlur(input_img, 3)
    equ = cv.equalizeHist(median)

    kernel = np.array([[0.0, -1.0, 0.0],
                       [-1.0, 5.0, -1.0],
                       [0.0, -1.0, 0.0]])

    kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)

    img_rst = cv.filter2D(equ, -1, kernel)
    return img_rst

# Show the final result
def show_result(y_test, y_prd_test):
    print("classification_report")
    print(sm.classification_report(y_test, y_prd_test))
    print("precision_score")
    print(precision_score(y_test, y_prd_test, average='macro'))
    print("roc_auc_score")
    print(roc_auc_score(y_test, y_prd_test))
    print("recall_score")
    print(metrics.recall_score(y_test, y_prd_test, average='micro'))

train_urls = search_files("./Plant_Phenotyping_Datasets/Plant/Ara2013-Canon","./Plant_Phenotyping_Datasets/Plant/Tobacco/",["Arabidopsis","tobacco"])
#print(train_urls)

start = time.time()
train_x, train_y = [], []
for label, urls in train_urls.items():
    for file in urls:
        image = cv.imread(file, 0)
        h, w = image.shape
        f = 100 / min(h, w)
        image = cv.resize(image, None, fx=f, fy=f)
        image = normalize(image)
        image = preprocess(image)
        shift = cv.xfeatures2d.SIFT_create()
        points = shift.detect(image)
        _, desc = shift.compute(image, points)

        sample = np.mean(desc, axis=0)
        train_x.append(sample)
        train_y.append(label)
train_x = np.array(train_x)

#print(train_x[0],train_y[0])

encoder = sp.LabelEncoder()
train_y_label = encoder.fit_transform(train_y)
#print(train_x[0], train_y_label[0])

X_train,X_test,y_train,y_test = train_test_split(train_x,train_y_label,test_size=0.3)

#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

print("SVM")
svm_clf = svm.SVC(kernel="linear", degree=2, gamma="auto", probability=True)
svm_clf.fit(X_train, y_train)
y_prd_test = svm_clf.predict(X_test)
show_result(y_test, y_prd_test)


print("MNB")
MNB_clf = MultinomialNB(alpha=0.01)
MNB_clf.fit(X_train, y_train)
y_prd_test = MNB_clf.predict(X_test)
show_result(y_test, y_prd_test)
end = time.time()
print("\ntime")
print(end-start)