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
    NB_perf = {}
    LDA_perf = {}
    SVM_perf = {}
    for n_pca in [10, 15, 20, 25, 30]:
        model_pca = PCA(n_components = n_pca)
        train_pca = model_pca.fit_transform(wbcd_train.data)
        test_pca = model_pca.transform(wbcd_test.data)

        model_NB = MultinomialNB().fit(np.absolute(train_pca), wbcd_train.target)
        NB_prediction = model_NB.predict(np.absolute(test_pca))
        scr_acc = accuracy_score(wbcd_test.target, NB_prediction)
        scr_pre = precision_score(wbcd_test.target, NB_prediction, average='macro')
        NB_perf[n_pca] = [round(scr_acc, 3), round(scr_pre, 3)]

        model_SVM = SVC(kernel='rbf').fit(train_pca[:400], wbcd_train.target[:400])
        SVM_prediction = model_SVM.predict(test_pca[201:])
        scr_acc = accuracy_score(wbcd_test.target[201:], SVM_prediction)
        scr_pre = precision_score(wbcd_test.target[201:], SVM_prediction, average='macro')
        SVM_perf[n_pca] = [round(scr_acc, 3), round(scr_pre, 3)]

        model_LDA = LinearDiscriminantAnalysis().fit(train_pca, wbcd_train.target)
        LDA_prediction = model_LDA.predict(test_pca)
        scr_acc = accuracy_score(wbcd_test.target, LDA_prediction)
        scr_pre = precision_score(wbcd_test.target, LDA_prediction, average='macro')
        LDA_perf[n_pca] = [round(scr_acc, 3), round(scr_pre, 3)]

    print(NB_perf)
    print(SVM_perf)
    print(LDA_perf)

    print("Model : Naive Bayes")
    table = [["10", NB_perf[10][0], NB_perf[10][1]],["15", NB_perf[15][0], NB_perf[15][1]],["20", NB_perf[20][0], NB_perf[20][1]],["25", NB_perf[25][0], NB_perf[25][1]], ["30", NB_perf[30][0], NB_perf[30][1]]]
    heads = ["No. of PCs", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))

    print("Model : SVM")
    table = [["10", SVM_perf[10][0], SVM_perf[10][1]],["15", SVM_perf[15][0], SVM_perf[15][1]],["20", SVM_perf[20][0], SVM_perf[20][1]],["25", SVM_perf[25][0], SVM_perf[25][1]], ["30", SVM_perf[30][0], SVM_perf[30][1]]]
    heads = ["No. of PCs", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))

    print("Model : LDA")
    table = [["10", LDA_perf[10][0], LDA_perf[10][1]],["15", LDA_perf[15][0], LDA_perf[15][1]],["20", LDA_perf[20][0], LDA_perf[20][1]],["25", LDA_perf[25][0], LDA_perf[25][1]], ["30", LDA_perf[30][0], LDA_perf[30][1]]]
    heads = ["No. of PCs", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))
