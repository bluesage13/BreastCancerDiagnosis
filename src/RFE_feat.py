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
from tabulate import tabulate

if __name__ == "__main__":
    wbcd_train = load_breast_cancer()
    wbcd_test = wbcd_train

    NB_perf = {}
    for n_feat in [5, 10, 15, 20, 25]:
        model_NB = MultinomialNB()
        model_rfe = RFE(model_NB, n_feat, step=1)
        model_rfe = model_rfe.fit(wbcd_train.data, wbcd_train.target)

        selected_features_to_drop = []
        for i in range (0, len(model_rfe.support_)):
            if(model_rfe.support_[i] == False):
                selected_features_to_drop.append(wbcd_train.feature_names[i])

        temp_dataset = load_breast_cancer()
        temp_dataFrame = pd.DataFrame(temp_dataset.data, columns=temp_dataset.feature_names)
        temp_dataFrame = temp_dataFrame.drop(selected_features_to_drop, axis=1)
        temp_dataset = Bunch(data=temp_dataFrame.values, target=wbcd_train.target, target_names=wbcd_train.target_names, feature_names = temp_dataFrame.columns)

        #print("Training Naive Bayes Model")
        model_NB = MultinomialNB().fit(temp_dataset.data, temp_dataset.target)
        NB_prediction = model_NB.predict(temp_dataset.data)
        scr_acc = accuracy_score(temp_dataset.target, NB_prediction)
        scr_pre = precision_score(temp_dataset.target, NB_prediction, average='macro')
        #print("Accuracy : " + str(round(scr_acc, 3)))
        #print("Precision : " + str(round(scr_pre, 3)))
        NB_perf[n_feat] = [round(scr_acc, 3), round(scr_pre, 3)]

    print(NB_perf)


    #SVM RFE
    SVM_perf = {}
    for n_feat in [5, 10, 15, 20, 25]:
        model_SVM = LinearSVC()
        model_rfe = RFE(model_SVM, n_feat, step=1)
        model_rfe = model_rfe.fit(wbcd_train.data, wbcd_train.target)

        selected_features_to_drop = []
        #print("The selected features are:")
        for i in range (0, len(model_rfe.support_)):
            if(model_rfe.support_[i] == False):
                selected_features_to_drop.append(wbcd_train.feature_names[i])

        #print(selected_features_to_drop)

        temp_dataset = load_breast_cancer()
        temp_dataFrame = pd.DataFrame(temp_dataset.data, columns=temp_dataset.feature_names)
        temp_dataFrame = temp_dataFrame.drop(selected_features_to_drop, axis=1)
        temp_dataset = Bunch(data=temp_dataFrame.values, target=wbcd_train.target, target_names=wbcd_train.target_names, feature_names = temp_dataFrame.columns)
        #print(temp_dataset.data.shape)

        #print("Training SVM model with RBF kernel function")
        model_SVM = SVC(kernel='rbf').fit(temp_dataset.data[:400], temp_dataset.target[:400])
        SVM_prediction = model_SVM.predict(temp_dataset.data[201:])
        scr_acc = accuracy_score(temp_dataset.target[201:], SVM_prediction)
        scr_pre = precision_score(temp_dataset.target[201:], SVM_prediction, average='macro')
        #print("Accuracy : " + str(round(scr_acc, 3)))
        #print("Precision : " + str(round(scr_pre, 3)))
        SVM_perf[n_feat] = [round(scr_acc, 3), round(scr_pre, 3)]

    print(SVM_perf)

    #LDA RFE
    LDA_perf = {}
    for n_feat in [5, 10, 15, 20, 25]:
        model_LDA = LinearDiscriminantAnalysis()
        model_rfe = RFE(model_LDA, n_feat, step=1)
        model_rfe = model_rfe.fit(wbcd_train.data, wbcd_train.target)

        selected_features_to_drop = []
        #print("The selected features are:")
        for i in range (0, len(model_rfe.support_)):
            if(model_rfe.support_[i] == False):
                selected_features_to_drop.append(wbcd_train.feature_names[i])

        #print(selected_features_to_drop)

        temp_dataset = load_breast_cancer()
        temp_dataFrame = pd.DataFrame(temp_dataset.data, columns=temp_dataset.feature_names)
        temp_dataFrame = temp_dataFrame.drop(selected_features_to_drop, axis=1)
        temp_dataset = Bunch(data=temp_dataFrame.values, target=wbcd_train.target, target_names=wbcd_train.target_names, feature_names = temp_dataFrame.columns)
        #print(temp_dataset.data.shape)

        #print("Training SVM model with RBF kernel function")
        model_LDA = LinearDiscriminantAnalysis().fit(temp_dataset.data, temp_dataset.target)
        LDA_prediction = model_LDA.predict(temp_dataset.data)
        scr_acc = accuracy_score(temp_dataset.target, LDA_prediction)
        scr_pre = precision_score(temp_dataset.target, LDA_prediction, average='macro')
        #print("Accuracy : " + str(round(scr_acc, 3)))
        #print("Precision : " + str(round(scr_pre, 3)))
        LDA_perf[n_feat] = [round(scr_acc, 3), round(scr_pre, 3)]

    print(LDA_perf)
    print("Model : Naive Bayes")
    table = [["5", NB_perf[5][0], NB_perf[5][1]],["10", NB_perf[10][0], NB_perf[10][1]],["15", NB_perf[15][0], NB_perf[15][1]],["20", NB_perf[20][0], NB_perf[20][1]], ["25", NB_perf[25][0], NB_perf[25][1]]]
    heads = ["No. of features", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))

    print("Model : SVM")
    table = [["5", SVM_perf[5][0], SVM_perf[5][1]],["10", SVM_perf[10][0], SVM_perf[10][1]],["15", SVM_perf[15][0], SVM_perf[15][1]],["20", SVM_perf[20][0], SVM_perf[20][1]], ["25", SVM_perf[25][0], SVM_perf[25][1]]]
    heads = ["No. of features", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))

    print("Model : LDA")
    table = [["5", LDA_perf[5][0], LDA_perf[5][1]],["10", LDA_perf[10][0], LDA_perf[10][1]],["15", LDA_perf[15][0], LDA_perf[15][1]],["20", LDA_perf[20][0], LDA_perf[20][1]], ["25", LDA_perf[25][0], LDA_perf[25][1]]]
    heads = ["No. of features", "Accuracy", "Precision"]
    print(tabulate(table, heads, tablefmt="grid"))
