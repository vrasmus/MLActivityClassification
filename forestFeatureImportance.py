#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:57:51 2017

@author: rv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split


from exploration import loadSubject,slidingWindows
from summarizingData import summarizeWindows

"""----------------------------------------------------------------------------
-------------------------------------------------------------------------------
----------------------------------------------------------------------------"""

nEstimators_ = 25
data = loadSubject(101)
nTopFeatures = 12

MODE = 'windows'
wsize=8

"""----------------------------------------------------------------------------
-------------------------------------------------------------------------------
----------------------------------------------------------------------------"""

nCV = 10

#wsizes = np.array(wsizes)
#score_mean = np.zeros(len(wsize))
#score_std = np.zeros(len(wsize))

if MODE == 'summaries':
    data_w = summarizeWindows(data,wsize)
elif MODE == 'windows':
    data_w = slidingWindows(data,wsize,step=1)

#Dropping time as a prediction parameter
data_w = data_w.drop('time',axis=1)

data_w = data_w.iloc[np.random.permutation(len(data_w))]

X = data_w.drop('activity',axis=1)
y = data_w['activity']        

clf = RandomForestClassifier(n_estimators=nEstimators_)

clf.fit(X,y)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

importances = importances[indices]
importances = importances[:nTopFeatures]
std = std[indices]
std = std[:nTopFeatures]

from pylab import rcParams
rcParams['figure.figsize'] = 6, 3
rcParams['axes.labelsize'] = 8
rcParams['axes.titlesize'] = 11
rcParams['axes.grid'] = 'off'
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['font.size'] = 7

fig, ax = plt.subplots()
plt.title("Feature importances")
plt.bar(range(len(importances)), importances, yerr=std, capsize=4,align="center")
plt.xticks(range(len(importances)), X.columns.unique()[indices][:nTopFeatures])
plt.ylim([0,np.max(std+importances)*1.1])
plt.xlim([-1, len(importances)])
fig.autofmt_xdate()
plt.savefig('fig.pdf',bbox_inches='tight', pad_inches=0)