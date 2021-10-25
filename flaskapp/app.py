import flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math
import os
import sys
import pandas as pd
from glob import glob
app = Flask(__name__)
pca_model = pickle.load(open('lda.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

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


@app.route('/')
def home():
    return render_template('index.html')

app.config["FILE_UPLOADS"] = r"static/Files"

@app.route('/uploader',methods=['GET','POST'])
def uploader():
    if(request.method == 'POST'):
        filepath = r'static/Files'
        filelist = [ fil for fil in os.listdir(filepath)]
        for fil in filelist:
            os.remove(os.path.join(filepath, fil))
        retrievedfile = request.files['myfile']
        retrievedfile.save(os.path.join(app.config["FILE_UPLOADS"],retrievedfile.filename))
        filepath = r'static/Files'
        print("File saved")
        for retfile in os.listdir(filepath):
            ret_file = retfile
            with open(os.path.join(filepath, ret_file), encoding='unicode_escape') as textfile:
                text=textfile.read()
                analyzer=BFAnalyzer()
                signature=analyzer.compute(text).compand().normalize().frequency()
            break
            

        pickled_LDA=pickle.load(open('lda.pkl','rb'))
        xtest=np.array(pickled_LDA.transform([signature]))
        pickled_svm=pickle.load(open('model.pkl','rb'))
        pred=pickled_svm.predict(np.array(xtest))
        labels=['mak', 'csproj', 'rexx', 'jenkinsfile', 'ml', 'kt']
        predLabel=labels[pred[-1]-1]
        print(predLabel)
        return render_template('index.html', prediction_text='predicted File Type: {}'.format(predLabel))


if __name__ == "__main__":
    app.run(debug=True)
