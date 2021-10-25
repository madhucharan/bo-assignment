import math
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV


SIGMA = 0.0035
SIGMA_SQR = SIGMA * SIGMA
class BFAnalyzer:
    def __init__(self):
        self.__frequency = [float(0)]*256#byteFrequencyarray
    def compute(self, text):
        for i in range(256):
            self.__frequency[i]=text.count(chr(i))#frequencyCalculation
        return self
    def normalize(self):
        try:
            norm = max(self.__frequency)
            self.__frequency =  list(map(lambda x: x / norm, self.__frequency))#Normalizing data
        except:
            pass
        return self
    def compand(self):
        B = 1.5
        self.__frequency =  list(map(lambda x: x ** ( 1 / B), self.__frequency))#Comapnding data
        return self
    def frequency(self):
        return np.array(self.__frequency)
    def __str__(self):
        return ",".join(map(str, self.__frequency))

class BFFileprint:
    def __init__(self,signatures):
        self.__fileprint=[float(0)]*256
        self.__signatures=signatures
    def computeFileprint(self):
        tot=len(self.__signatures)
        for i in range(256):
            self.__fileprint[i]=(sum([x[i] for x in self.__signatures])/tot)#Averaging all signatures for filePrint
        return self
    def fileprint(self):
        return self.__fileprint
    
#ByteFrequencyCorrelator calucates correlation for each file with corresponding fileType Fingerprint
class ByteFrequencyCorrelator:
    def __init__(self,filePrint):
        self.filePrint = filePrint
    def correlate(self, signature):
        self.cmpSignature = signature
        self.correlation = [None] * 256
        for i in range(256):
            diff = self.cmpSignature[i] - self.baseSignature[i]
            exp = ( -1 * diff * diff ) / ( 2 * SIGMA_SQR )
            self.correlation[i] = math.exp(exp)
        return self.correlation

#ByteFrequencyCrossCorrelator calculates correlation for every byte with every other byte
class BFCrossCorrelator:
    def __init__(self, baseSignature):
        self.baseSignature = baseSignature
    def correlate(self):
        self.correlation = Matrix = [[0 for x in range(256)] for x in range(256)]
        for i in range(256):
            for j in range(i):
                freqDiff = ( self.baseSignature[i] - self.baseSignature[j] )
                exp = ( -1 * freqDiff * freqDiff ) / ( 2 * SIGMA_SQR )
                self.correlation[i][j] = freqDiff
                self.correlation[j][i] = math.exp(exp)
        return self.correlation


class Models:
    def __init__(self) -> None:
        self.Xtrain =[]
        self.Xtest =[]
        self.sytrain  =[]
        self.ytest = []

    def preprocess(self,data):
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


    

class FeatureSelection:
    def __init__(self) -> None:
        pass
    def apply_pca(self, df):
        pca=PCA(n_components=3)
        comps=pca.fit_transform(df)
        PCADF = pd.DataFrame(data =comps,columns = ['pc1', 'pc2','pc3'])

    def apply_kernalpca(self,df):
        kpca = KernelPCA(kernel="rbf", gamma=15, n_components=3)
        kcomps=kpca.fit_transform(df)
        KPCADF = pd.DataFrame(data =kcomps,columns = ['pc1', 'pc2','pc3'])

    def apply_lda(self,df,numy):
        lda=LinearDiscriminantAnalysis(n_components=3)
        comps=lda.fit(df,numy).transform(df)
        LDADF = pd.DataFrame(data =comps,columns = ['pc1', 'pc2','pc3'])    

    def apply_nmf(self,df):
        nmf=NMF(n_components=3,random_state=1)
        comps=nmf.fit_transform(df)
        NMFDF = pd.DataFrame(data =comps,columns = ['pc1', 'pc2','pc3'])

    def plot_dist(self,df,numy):
        ax = plt.axes(projection='3d')
        ax.scatter3D(df['pc3'],df['pc1'],df['pc2'], c=numy, cmap='gist_rainbow')



