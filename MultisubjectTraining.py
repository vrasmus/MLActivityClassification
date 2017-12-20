#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 08:55:45 2017

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
#features = ['time','activity','hand_acc_16g_x','hand_acc_16g_y','hand_acc_16g_z','hand_gyro_x','hand_gyro_y','hand_gyro_z','hand_magn_x','hand_magn_y','hand_magn_z']

""" Choose which configuration to use by commenting in."""

### This mode uses summary statistics, rolling mean and std.
#MODE = 'summaries'
wsize = 120
nEsts = 25

"""----------------------------------------------------------------------------
-------------------------------------------------------------------------------
----------------------------------------------------------------------------"""

Results = []
for testSub in Subs:
    print('Test Subject '+str(testSub))
    trainSubs = Subs[:]
    trainSubs.remove(testSub)
        
    init = 1
    for sub in trainSubs:
        print('Loading Subject '+str(sub))
        data = loadSubject(sub)
#        data = data[features]
        data_w = summarizeWindows(data,wsize)
    
        #Dropping time as a prediction parameter
        data_w = data_w.drop('time',axis=1)
        
        data_w = data_w.iloc[np.random.permutation(len(data_w))]
        
        X_ = data_w.drop('activity',axis=1)
        y_ = data_w['activity']
        
        if init==1:
            X_train = X_
            y_train = y_
            init = 0
        else:
            X_train = pd.concat([X_train,X_])
            y_train = pd.concat([y_train,y_])
        
    clf = RandomForestClassifier(n_estimators=nEsts)
    clf.fit(X_train,y_train)
    
    data = loadSubject(testSub)
#    data = data[features]
    data_w = summarizeWindows(data,wsize)
    #Dropping time as a prediction parameter
    data_w = data_w.drop('time',axis=1)
    
    data_w = data_w.iloc[np.random.permutation(len(data_w))]
        
    X_test = data_w.drop('activity',axis=1)
    y_test = data_w['activity']
    
    y_pred = clf.predict(X_test)
    resSub = accuracy_score(y_test,y_pred)
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
width = 0.50

fig, ax = plt.subplots()
ax.bar(ind,Results,width,color=('tab:blue','tab:orange','tab:green','tab:red'))
#for i,res in enumerate(Results):
#    places = [i + width*_ind for _ind in ind]
#    ax.bar(ind+(i*width), res, width,label='Subject '+str(Subs[i]))

ax.set_xticks(ind)
ax.set_xticklabels(('101','102','105','108'))
ax.set_xlabel('Test Subject')
ax.set_ylabel('Classification accuracy')
#ax.set_title('Cross-subject classification, hand accelerometer')
#ax.set_title('Cross-subject classification, heart rate + temperatures')
ax.set_title('Cross-subject classification, all features')
plt.legend()
plt.ylim([0,0.85])
plt.savefig('fig.pdf',bbox_inches='tight', pad_inches=0)