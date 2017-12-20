#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:54:15 2017

@author: rv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.model_selection as skmod
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score

from exploration import loadSubject,slidingWindows
from summarizingData import summarizeWindows

"""----------------------------------------------------------------------------
-------------------------------------------------------------------------------
----------------------------------------------------------------------------"""

Depths_ = [1,3,9,15,None]

data = loadSubject(101)

""" Choose which features to use. Outcomment all for all """
#data = data[['time','activity','hand_acc_16g_x','hand_acc_16g_y','hand_acc_16g_z']] #TRY ONLY WITH ACC HAND
#data = data[['time','activity','heart_rate','hand_acc_16g_x','hand_acc_16g_y','hand_acc_16g_z']] #TRY ONLY WITH ACC HAND+HEART RATE
#data = data[['time','activity','hand_temp','hand_acc_16g_x','hand_acc_16g_y','hand_acc_16g_z']] #TRY ONLY WITH HAND MEASUREMENTS
data = data[['time','activity','heart_rate','hand_temp','chest_temp','ankle_temp']]
  
""" Choose which configuration to use by commenting in."""

### This mode uses summary statistics, rolling mean and std.
#MODE = 'summaries'
#wsizes = [10,20,40,80,120]

### This mode allows prediction using full data for a given window.
MODE = 'windows'
wsizes = [1, 2, 4, 8, 16, 32]
steps = np.ones_like(wsizes)


"""----------------------------------------------------------------------------
-------------------------------------------------------------------------------
----------------------------------------------------------------------------"""

nCV = 10

wsizes = np.array(wsizes)
score_mean = np.zeros(len(wsizes))
score_std = np.zeros(len(wsizes))

from pylab import rcParams
rcParams['figure.figsize'] = 4, 3
rcParams['axes.labelsize'] = 8
rcParams['axes.titlesize'] = 11
rcParams['axes.grid'] = 'on'
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['font.size'] = 7

fig, ax = plt.subplots()
plt.ylim([0,1.05])

for depth in Depths_:
    for i, size in enumerate(wsizes):
        
        if MODE == 'summaries':
            data_w = summarizeWindows(data,size)
        elif MODE == 'windows':
            data_w = slidingWindows(data,size,step=steps[i])
    
        #Dropping time as a prediction parameter
        data_w = data_w.drop('time',axis=1)
        
        data_w = data_w.iloc[np.random.permutation(len(data_w))]
        
        X = data_w.drop('activity',axis=1)
        y = data_w['activity']
        
        clf = DecisionTreeClassifier(max_depth=depth)
    
        scores = cross_val_score(clf, X, y, cv=nCV)
    
        print(str(depth)+": Window size:",size,"Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        score_mean[i] = scores.mean()
        score_std[i] = scores.std()

    errDown = [2*j if score_mean[i] - 2*j > 0 else score_mean[i] for i,j in enumerate(score_std)]
    errUp = [2*j if score_mean[i] + 2*j < 1 else 1-score_mean[i] for i,j in enumerate(score_std)]
    plt.errorbar(wsizes,score_mean,[errDown, errUp],fmt='x-',capsize=4,label='Max depth ' + str(depth))

plt.legend()
ax.set_xlabel('Window size')
ax.set_ylabel('Accuracy')
#ax.set_title('Trees from Summary Statistics')
ax.set_title('Trees from Raw Data Windows')
plt.savefig('fig.pdf',bbox_inches='tight', pad_inches=0)
