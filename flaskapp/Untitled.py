import math
import os
import sys
import numpy as np
import pandas as pd
import pickle
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
    
path='/home/surya/blueOptima/worksample_data/kt/1.kt'
with open(path) as textfile:
    text=textfile.read()
    analyzer=BFAnalyzer()
    signature=analyzer.compute(text).compand().normalize().frequency()

pickled_LDA=pickle.load(open('lda.pkl','rb'))
xtest=np.array(pickled_LDA.transform([signature]))
pickled_svm=pickle.load(open('model.pkl','rb'))
pred=pickled_svm.predict(np.array(xtest))
labels=['mak', 'csproj', 'rexx', 'jenkinsfile', 'ml', 'kt']
predLabel=labels[pred[-1]-1]
print('predicted fileType : '+predLabel)
