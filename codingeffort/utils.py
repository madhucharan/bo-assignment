import math
import sklearn
import matplotlib
import seaborn as sns
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

        

SIGMA = 0.0035
SIGMA_SQR = SIGMA * SIGMA

class BFAnalyzer:
    def __init__(self):
        self.__frequency = [float(0)]*256
    def compute(self, text):
        for i in range(256):
            self.__frequency[i]=text.count(chr(i))
        return self
    def normalize(self):
        try:
            norm = max(self.__frequency)
            self.__frequency =  list(map(lambda x: x / norm, self.__frequency))
        except:
            pass
        return self
    def compand(self):
        B = 1.5
        self.__frequency =  list(map(lambda x: x ** ( 1 / B), self.__frequency))
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
            self.__fileprint[i]=(sum([x[i] for x in self.__signatures])/tot)
        return self
    def fileprint(self):
        return self.__fileprint

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

