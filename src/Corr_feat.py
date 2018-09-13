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
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from tabulate import tabulate

if __name__ == "__main__":
    wbcd_train = load_breast_cancer()
    NB_perf = {}
    LDA_perf = {}
    SVM_perf = {}

    for corr_val in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        wbcd_df = pd.DataFrame(wbcd_train.data, columns=wbcd_train.feature_names)
        cm_temp = wbcd_df.corr()
        corr_mat = cm_temp.abs()
        triange_up = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
        dropped_columns = [col for col in triange_up.columns if any(triange_up[col] > corr_val)]
        wbcd_df_pruned = wbcd_df.drop(dropped_columns, axis=1)
        wbcd_train_pruned = Bunch(data=wbcd_df_pruned.values, target = wbcd_train.target, target_names=wbcd_train.target_names, feature_names = wbcd_df_pruned.columns)
        wbcd_test = wbcd_train_pruned

        model_NB = MultinomialNB().fit(wbcd_train_pruned.data, wbcd_train_pruned.target)
        NB_prediction = model_NB.predict(wbcd_test.data)
        scr_acc = accuracy_score(wbcd_test.target, NB_prediction)
        scr_pre = precision_score(wbcd_test.target, NB_prediction, average='macro')
        NB_perf[corr_val] = [round(scr_acc, 3), round(scr_pre, 3)]

        model_SVM = SVC(kernel='rbf').fit(wbcd_train_pruned.data[:400], wbcd_train_pruned.target[:400])
        SVM_prediction = model_SVM.predict(wbcd_test.data[201:])
        scr_acc = accuracy_score(wbcd_test.target[201:], SVM_prediction)
        scr_pre = precision_score(wbcd_test.target[201:], SVM_prediction, average='macro')
        SVM_perf[corr_val] = [round(scr_acc, 3), round(scr_pre, 3)]

        model_LDA = LinearDiscriminantAnalysis().fit(wbcd_train_pruned.data, wbcd_train_pruned.target)
        LDA_prediction = model_LDA.predict(wbcd_test.data)
        scr_acc = accuracy_score(wbcd_test.target, LDA_prediction)
        scr_pre = precision_score(wbcd_test.target, LDA_prediction, average='macro')
        LDA_perf[corr_val] = [round(scr_acc, 3), round(scr_pre, 3)]

    print(NB_perf)
    print(SVM_perf)
    print(LDA_perf)
    print("Model : Naive Bayes")
    table = [["0.7 >", NB_perf[0.7][0], NB_perf[0.7][1]],["0.75 > ", NB_perf[0.75][0], NB_perf[0.75][1]],["0.8 >", NB_perf[0.8][0], NB_perf[0.8][1]],["0.85 >", NB_perf[0.85][0], NB_perf[0.85][1]], ["0.9 >", NB_perf[0.9][0], NB_perf[0.9][1]], ["0.95 >", NB_perf[0.95][0], NB_perf[0.95][1]]]
    heads = ["No. of features", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))

    print("Model : SVM")
    table = [["0.7 >", SVM_perf[0.7][0], SVM_perf[0.7][1]],["0.75 >", SVM_perf[0.75][0], SVM_perf[0.75][1]],["0.8 >", SVM_perf[0.8][0], SVM_perf[0.8][1]],["0.85 >", SVM_perf[0.85][0], SVM_perf[0.85][1]], ["0.9 >", SVM_perf[0.9][0], SVM_perf[0.9][1]],["0.95 >", SVM_perf[0.95][0], SVM_perf[0.95][1]]]
    heads = ["No. of features", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))

    print("Model : LDA")
    table = [["0.7 >", LDA_perf[0.7][0], LDA_perf[0.7][1]],["0.75 >", LDA_perf[0.75][0], LDA_perf[0.75][1]],["0.8 >", LDA_perf[0.8][0], LDA_perf[0.8][1]],["0.85 >", LDA_perf[0.85][0], LDA_perf[0.85][1]], ["0.9 >", LDA_perf[0.9][0], LDA_perf[0.9][1]], ["0.95 >", LDA_perf[0.95][0], LDA_perf[0.95][1]]]
    heads = ["Correlation", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))
