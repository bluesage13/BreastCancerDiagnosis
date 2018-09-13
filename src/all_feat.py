import numpy as np
from sklearn.datasets.base import Bunch
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from tabulate import tabulate

if __name__ == "__main__":
    wbcd_train = load_breast_cancer()
    wbcd_test = wbcd_train
    perf = {}
    #print("Training Naive Bayes Model")
    model_NB = MultinomialNB().fit(wbcd_train.data, wbcd_train.target)
    NB_prediction = model_NB.predict(wbcd_test.data)
    scr_acc = accuracy_score(wbcd_test.target, NB_prediction)
    scr_pre = precision_score(wbcd_test.target, NB_prediction, average='macro')
    #print("Accuracy : " + str(round(scr_acc, 3)))
    #print("Precision : " + str(round(scr_pre, 3)))
    perf['NB'] = [round(scr_acc, 3), round(scr_pre, 3)]
    #print("Training SVM model with RBF kernel function")
    model_SVM = SVC(kernel='rbf').fit(wbcd_train.data[:400], wbcd_train.target[:400])
    SVM_prediction = model_SVM.predict(wbcd_test.data[201:])
    scr_acc = accuracy_score(wbcd_test.target[201:], SVM_prediction)
    scr_pre = precision_score(wbcd_test.target[201:], SVM_prediction, average='macro')
    #print("Accuracy : " + str(round(scr_acc, 3)))
    #print("Precision : " + str(round(scr_pre, 3)))
    perf['SVM'] = [round(scr_acc, 3), round(scr_pre, 3)]
    #print("Training LDA model")
    model_LDA = LinearDiscriminantAnalysis().fit(wbcd_train.data, wbcd_train.target)
    LDA_prediction = model_LDA.predict(wbcd_test.data)
    scr_acc = accuracy_score(wbcd_test.target, LDA_prediction)
    scr_pre = precision_score(wbcd_test.target, LDA_prediction, average='macro')
    #print("Accuracy : " + str(round(scr_acc, 3)))
    #print("Precision : " + str(round(scr_pre, 3)))
    perf['LDA'] = [round(scr_acc, 3), round(scr_pre, 3)]

    table = [["Naive Bayes", perf['NB'][0], perf['NB'][1]], ["SVM", perf['SVM'][0], perf['SVM'][1]], ["LDA", perf['LDA'][0], perf['LDA'][1]]]
    heads = ["Models", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))
