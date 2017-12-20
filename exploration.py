#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:23:02 2017

@author: rv

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4512690/#pone.0130851.ref029
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def loadSubject(num,interpolate=True):
    datafile = '~/Desktop/PAMAP2_Dataset/Protocol/subject' + str(num) + '.dat'
    data = pd.read_csv(datafile, header= None, sep=" ")
    
    # Give columns names
    sensorMeasurements = ['temp','acc_16g_x','acc_16g_y','acc_16g_z','acc_6g_x','acc_6g_y','acc_6g_z','gyro_x','gyro_y','gyro_z','magn_x','magn_y','magn_z','orien_1','orien_2','orien_3','orien_4']
    allSensors = [spot+measurement for spot in ['hand_','chest_','ankle_'] for measurement in sensorMeasurements ]
    data.columns = ['time','activity','heart_rate'] + allSensors
    if interpolate==True:
        data['heart_rate'] = data['heart_rate'].interpolate()

    
    # Other data not really useful, but large amount. Discard...
    data = data[data['activity'] != 0]
    #Dropping columns that are not useful according to readme.
    data = data.drop([spot+measurement for spot in ['hand_','chest_','ankle_'] for measurement in ['orien_1','orien_2','orien_3','orien_4']],axis=1)
    #Dropping colums with the low-precision accelerometer, since should be highly correlated with the other anyway.
    data = data.drop([spot+measurement for spot in ['hand_','chest_','ankle_'] for measurement in ['acc_6g_x','acc_6g_y','acc_6g_z']],axis=1)    
    
    #Rename activity
    mapActivity = {1 :'lying', 2:'sitting',3: 'standing', 4: 'walking',5: 'running',\
                   6: 'cycling',7: 'Nordic walking',9: 'watching TV',10: 'computer work',\
                   11: 'car driving',12: 'ascending stairs',13: 'descending stairs',\
                   16: 'vacuum cleaning',17: 'ironing',18: 'folding laundry',\
                   19: 'house cleaning',20: 'playing soccer',24: 'rope jumping',0: 'other'}
    dataReplacer = {'activity':mapActivity}
    data = data.replace(dataReplacer)
    
    ##Note : There is about 1% of rows still containing NANS after interpolation.
    return data

def slidingWindows(data,windowSize,step=1):
    stripped = data.drop(['time','activity'],axis=1)
    names = stripped.columns.unique()
        
    init = 1
    for act in data['activity'].unique():
        data_act = data.loc[data['activity'] == act]
            
        data_ = data_act[:-windowSize:step]
        data_ = data_.set_index(np.arange(len(data_)))
        
        for i in np.arange(1,windowSize):
            numberedColumns = [name + '_'+str(i) for name in names]
            stripped.columns = numberedColumns
#            new_ = stripped[i:i-windowSize:step]
#            new_ = stripped[i::step]
#            new_ = new_.set_index(np.arange(len(new_)))
#            new_ = new_.set_index(np.arange(1+i,1+i+len(new_)))
            new_ = stripped.set_index(np.arange(i,i+len(stripped)))
            data_ = pd.concat([data_,new_],axis=1)
    
        data_ = data_.dropna()        
            
        if init == 1:
            res = data_
            init = 0
        else:
            res = pd.concat([res,data_])
    return res

if __name__ == '__main__':
    subjects = [101,102,103,104,105,106,107,108,109]
#    subjects = [101,102,105]
    
    _init = 1
    for subject in subjects:
        data = loadSubject(subject)
        if _init==1:
            vc = data['activity'].value_counts()
            _init = 0
        elif _init==0:
            vc_ = data['activity'].value_counts()
            vc = pd.concat([vc,vc_],axis=1)
        
    vc.columns = ['Subject '+str(sub) for sub in subjects]    
    vc_order = vc.sum(axis=1).sort_values(ascending=False)
    vc = vc.reindex(vc_order.index)
    vc = vc/vc.sum().sum()
    
    from pylab import rcParams
    rcParams['figure.figsize'] = 6, 4
    rcParams['axes.labelsize'] = 8
    rcParams['axes.titlesize'] = 11
    rcParams['axes.grid'] = 'off'
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['font.size'] = 7
    
    plt.figure()
    ax = vc.plot(kind='bar',stacked=True)
    fig = ax.get_figure()
    fig.autofmt_xdate()
    
    ax.set_title('Classified instances')
    plt.savefig('fig.pdf',bbox_inches='tight', pad_inches=0)

    
#    ### Other data not really useful, but large amount. Discard...
#    data = data[data['activity'] != 'other']
#    
#    plt.figure()
#    vc = data['activity'].value_counts()
#    vc = vc/np.sum(vc)
#    ax = vc.plot(kind='bar')
#    fig = ax.get_figure()
#    fig.autofmt_xdate()
#    
#    #Dropping columns that are not useful according to readme.
#    data = data.drop([spot+measurement for spot in ['hand_','chest_','ankle_'] for measurement in ['orien_1','orien_2','orien_3','orien_4']],axis=1)
#    #Dropping colums with the low-precision accelerometer, since should be highly correlated with the other anyway.
#    data = data.drop([spot+measurement for spot in ['hand_','chest_','ankle_'] for measurement in ['acc_6g_x','acc_6g_y','acc_6g_z']],axis=1)
#    
#    
#    invMapActivity = {v: k for k, v in mapActivity.items()}
#    dataReplacer = {'activity':invMapActivity}
#    data = data.replace(dataReplacer)
#    
#    plt.figure()
#    plt.plot(data['time'],data['activity'],'.')
#    # Make sure that timestamp is removed when doing predictions..

"""
NOTES: 
Interpolate Heart Rate to be able to use it for single measurements.
Create 'concatenated' values for some time period to better predict.
See how good the prediction is with just one sensor (arm), to simulate smart-watch
"""