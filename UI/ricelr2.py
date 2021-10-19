from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
df=pd.read_csv('rice.csv')
df=df.iloc[:119]
train=df[:107]
test=df[107:]
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Rice']
X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Rice']
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
poly.fit(X_poly,y_train)
lr_2=LinearRegression()
lr_2.fit(X_poly,y_train)
pickle.dump(lr_2, open("rice1.pkl",'wb'))
rice=pickle.load('rice1.pkl','rb')