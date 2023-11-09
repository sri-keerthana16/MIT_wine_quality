import os
os.chdir('/home/srikeerthana/wine/')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer,StandardScaler
from sklearn import svm
data = pd.read_csv("winequality-red.csv")
X = data.iloc[:, :-1]
X = StandardScaler().fit_transform(X)
y = data.iloc[:, -1]
data['goodquality'] = [1 if x >= 7 else 0 for x in data['quality']]
X = data.drop(['quality','goodquality'], axis = 1)
y = data['goodquality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rndf=RandomForestClassifier(n_estimators=100, random_state=20).fit(X_train,y_train)
yperd1=rndf.predict(X_test)
import flask
from flask import Flask,jsonify,request
data=pd.read_csv("winequality-red.csv")
import numpy as np
app=Flask('Wine Quality Prediction')
@app.route('/hello')
def new():
    return "hello how are you !"
@app.route("/<float:fixed_acidity>/<float:volatile_acidity>/<float:citric_acid>/<float:residual_sugar>/<float:chlorides>/<float:free_sulfur_dioxide>/<float:total_sulfur_dioxide>/<float:density>/<float:pH>/<float:sulphates>/<float:alcohol>")
def home(fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol):
    p=[]
    p+=[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]
    arr=np.array([p])
    predict=rndf.predict(arr)
    if predict == [1]:
        result = {'result':'Good quality'}
    else:
        result ={'result':'Not Good quality'}
    return jsonify(result)
