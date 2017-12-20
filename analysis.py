#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:02:24 2017

@author: rv
"""

import numpy as np
from matplotlib import pyplot as plt
from exploration import loadSubject

data = loadSubject(101,interpolate=False)

desc = data.describe().T
desc = desc.round(2)
print(desc)

act_desc = []

for act in data['activity'].unique():
    data_act = data.loc[data['activity'] == act]
    
    desc_ = data_act.describe().T
    act_desc.append((act,desc_))


data_ = data.drop(['time','activity'],axis=1)
features = data_.columns.unique()
for feature in features:
    m = np.zeros(len(act_desc))
    s = np.zeros(len(act_desc))
    
    for i,x in enumerate(act_desc):
        act,d = x
        m[i] = act_desc[i][1].loc[feature][1]
        s[i] = act_desc[i][1].loc[feature][2]
    
    x = range(len(act_desc))
    inds = np.argsort(m)
    m = m[inds]
    s = s[inds]
    acts = [a[0] for a in act_desc]
    acts = np.array(acts)[inds]
    
    fig, ax = plt.subplots()
    ax.set_title(feature)
    ax.errorbar(x,m, yerr=2*s,capsize=4,fmt='x')
    plt.xticks(x,acts,rotation=75)