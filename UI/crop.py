from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

crop = pd.read_csv("croprec.csv")
crop.shape
df=crop
target = df['label']
features1 = df[['N','P','K','temperature', 'ph', 'rainfall']]
from sklearn.model_selection import train_test_split
Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(features1,target,test_size = 0.20,random_state =311)
b=Ytrain1
from sklearn.ensemble import VotingClassifier
em = VotingClassifier(estimators=[
       ('knn', KNeighborsClassifier(n_neighbors=3)), ('dt',DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)), ('gnb',GaussianNB() )],
       voting='hard')
em.fit(Xtrain1,Ytrain1)
pickle.dump(em, open("model1.pkl",'wb'))
model=pickle.load('model1.pkl','rb')