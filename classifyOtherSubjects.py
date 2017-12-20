#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:10:23 2017

@author: rv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


from exploration import loadSubject,slidingWindows
from summarizingData import summarizeWindows


"""----------------------------------------------------------------------------
-------------------------------------------------------------------------------
----------------------------------------------------------------------------"""

Subs = [101,102,105,108]

""" Choose which features to use. Outcomment all for all """
#features = ['time','activity','heart_rate','hand_temp','chest_temp','ankle_temp']
#features = ['time','activity','hand_acc_16g_x','hand_acc_16g_y','hand_acc_16g_z']
features = ['time','activity','hand_acc_16g_x','hand_acc_16g_y','hand_acc_16g_z','hand_gyro_x','hand_gyro_y','hand_gyro_z','hand_magn_x','hand_magn_y','hand_magn_z']

""" Choose which configuration to use by commenting in."""

### This mode uses summary statistics, rolling mean and std.
#MODE = 'summaries'
wsize = 120
nEsts = 25

"""----------------------------------------------------------------------------
-------------------------------------------------------------------------------
----------------------------------------------------------------------------"""

Results = []
for trainSub in Subs:
    print('Training Subject '+str(trainSub))
    testSubs = Subs[:]
    testSubs.remove(trainSub)
    
    data = loadSubject(trainSub)
    data = data[features]
    data_w = summarizeWindows(data,wsize)
    #Dropping time as a prediction parameter
    data_w = data_w.drop('time',axis=1)
    
    data_w = data_w.iloc[np.random.permutation(len(data_w))]
        
    X = data_w.drop('activity',axis=1)
    y = data_w['activity']
        
    clf = RandomForestClassifier(n_estimators=nEsts)
    clf.fit(X,y)
    resSub = []
    for sub in Subs:
        print('Testing Subject '+str(sub))
        data = loadSubject(sub)
        data = data[features]
        data_w = summarizeWindows(data,wsize)
    
        #Dropping time as a prediction parameter
        data_w = data_w.drop('time',axis=1)
        
        data_w = data_w.iloc[np.random.permutation(len(data_w))]
        
        X_ = data_w.drop('activity',axis=1)
        y_ = data_w['activity']
        
        y_pred = clf.predict(X_)
        if sub != trainSub:
            resSub.append(accuracy_score(y_,y_pred))
        else:
            resSub.append(0)
        print(accuracy_score(y_,y_pred))
    Results.append(resSub)
    
    
"""----------------------------------------------------------------------------
-------------------------------------------------------------------------------
----------------------------------------------------------------------------"""

from pylab import rcParams
rcParams['figure.figsize'] = 5, 3
rcParams['axes.labelsize'] = 8
rcParams['axes.titlesize'] = 11
rcParams['axes.grid'] = 'off'
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['font.size'] = 7

Results = np.array(Results).T
print('Average classification accuracy:', np.mean(Results[Results.nonzero()]))

ind = np.arange(len(Subs))
c = ['b','r','g','y']
width = 0.15

fig, ax = plt.subplots()
for i,res in enumerate(Results):
    places = [i + width*_ind for _ind in ind]
    ax.bar(ind+(i*width), res, width,label='Subject '+str(Subs[i]))

ax.set_xticks(ind + width*1.5)
ax.set_xticklabels(('101','102','105','108'))
ax.set_xlabel('Training Subject')
ax.set_ylabel('Classification accuracy')
#ax.set_title('Cross-subject classification, hand accelerometer')
#ax.set_title('Cross-subject classification, heart rate + temperatures')
#ax.set_title('Cross-subject classification, all features')
ax.set_title('Cross-subject classification')
plt.legend()
plt.ylim([0,0.6])
plt.savefig('fig.pdf',bbox_inches='tight', pad_inches=0)
