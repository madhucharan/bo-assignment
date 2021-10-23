
import os
import math
import sys
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV


class Models:
    def __init__(self) -> None:
        self.Xtrain =[]
        self.Xtest =[]
        self.sytrain  =[]
        self.ytest = []

    def preprocess(self,df):
        target= data['target']
        data= data.drop('target',axis=1)
        xtrain,xtest,self.ytrain,self.ytest = train_test_split(data,target,test_size=0.2)
        scaler = StandardScaler()
        self.Xtrain = scaler.fit_transform(xtrain)
        self.Xtest = scaler.transform(xtest)
        return self.Xtrain,self.Xtest,self.ytrain,self.ytest

    def svm_model(self):
        params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'],'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
        # Performing CV to tune parameters for best SVM fit 
        svm_model = GridSearchCV(SVC(), params_grid, cv=5)
        svm_model.fit(self.Xtrain, self.ytrain)

        return {'best_score':svm_model.best_score_,
                'best_estimator': svm_model.best_estimator_.C, 
                'best_kernel': svm_model.best_estimator_.kernel, 
                'gamma': svm_model.best_estimator_.gamma }

    
