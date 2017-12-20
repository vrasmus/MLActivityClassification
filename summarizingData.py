#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:05:36 2017

@author: rv
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from exploration import loadSubject


def summarizeWindows(data,windowSize,allowedMissing=2):
    stripped = data.drop(['time','activity'],axis=1)
    names = stripped.columns.unique()

#    res_m = pd.rolling_mean(stripped,windowSize,allowedMissing)
    res_m = stripped.rolling(window=windowSize,min_periods=2,center=False).mean()
#    res_s = pd.rolling_std(stripped,windowSize,allowedMissing)
    res_s = stripped.rolling(window=windowSize,min_periods=2,center=False).std()
     
    meanNamesColumns = [name + '_mean' for name in names]
    stdNamesColumns = [name + '_std' for name in names]
    
    res_m.columns = meanNamesColumns
    res_s.columns = stdNamesColumns
    
    res = pd.concat([data[['time','activity']],res_m],axis=1)
    res = pd.concat([res,res_s],axis=1)
    
    res = res.dropna()

    return res

if __name__ == '__main__':
    data = loadSubject(101)
    data = data[20000:30000]
    data_s = summarizeWindows(data,100)