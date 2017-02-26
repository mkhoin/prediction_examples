#-*- coding:utf-8 -*

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import BayesianRidge, LinearRegression 
from sklearn import svm

import matplotlib.pyplot as plt
import numpy as np

def get_dataset():
    fp = open('../dataset/boston.csv','r')
    title = fp.readline()
    Xdata = []
    Ydata = []
    for line in fp.readlines():
        line = list(map(float, line.strip().split(',')))
        Ydata.append(line[-1:])
        Xdata.append(line[:-1])
    return np.asarray(Xdata), np.asarray(Ydata)

if __name__ == "__main__":
    Xdata, Ydata = get_dataset()
    clf = svm.SVR()
    clf.fit(Xdata[:300],Ydata[:300])
    predicted = clf.predict(Xdata[-100:])
    #ols = LinearRegression()
    #clf = BayesianRidge(compute_score=True)
    
    #predicted = cross_val_predict(ols, Xdata, Ydata, cv=10)
    #predicted = cross_val_predict(clf, Xdata, Ydata, cv=10)
    
    fig, ax = plt.subplots()
    ax.scatter(Ydata[-100:], predicted)
    #ax.plot([Ydata.min(), Ydata.max()], [Ydata.min(), Ydata.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

