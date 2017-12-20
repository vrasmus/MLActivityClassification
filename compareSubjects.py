#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:56:16 2017

@author: rv
"""

 
import numpy as np
from matplotlib import pyplot as plt
from exploration import loadSubject

subs = [101,102,105,108]
data = [loadSubject(i,interpolate=False) for i in subs]

acts = []
for _,d in enumerate(data):
    for u in d['activity'].unique():
        if u not in acts:
            acts.append(u)
           
         
m_total = []
s_total = []

for _,d in enumerate(data):
    act_desc = []
    
    for act in acts:
        data_act = d.loc[d['activity'] == act]
        
        desc_ = data_act.describe().T
        act_desc.append((act,desc_))
        
    data_ = d.drop(['time','activity'],axis=1)
#    features = data_.columns.unique()
    features = ['heart_rate']
#    features = ['hand_acc_16g_x']
#    features = ['hand_temp']
#    features = ['hand_acc_16g_y']
    
    for feature in features:
        m = np.zeros(len(act_desc))
        s = np.zeros(len(act_desc))
        
        for i,x in enumerate(act_desc):
            act,d = x
            m[i] = act_desc[i][1].loc[feature][1]
            s[i] = act_desc[i][1].loc[feature][2]

        m_total.append((feature,_,m))
        s_total.append((feature,_,s))
        
"""---------------------------------------------------------------------------
---------------------------Make figure----------------------------------------
---------------------------------------------------------------------------"""

from pylab import rcParams
rcParams['figure.figsize'] = 4, 3
rcParams['axes.labelsize'] = 8
rcParams['axes.titlesize'] = 11
rcParams['axes.grid'] = 'on'
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['font.size'] = 7

x = np.arange(len(m_total[0][2]))
inds = np.argsort(m)
fig, ax = plt.subplots()

for i,_ in enumerate(m_total):
    m = m_total[i][2][inds]
    s = s_total[i][2][inds]
    acts = [a[0] for a in act_desc]
    acts = np.array(acts)[inds]
    
#    ax.set_title(feature)
    step_between = 0.1
    adj = step_between*len(m_total)/2 #Adjust to center around X 
    ax.errorbar(-adj+x+step_between*np.ones_like(x)*i,m, yerr=2*s,capsize=4,fmt='x',label='subject ' + str(subs[i]))
    plt.xticks(x,acts,rotation=75)
#    
ax.set_ylabel(r'BPM')
ax.set_title('Heart Rate')
#ax.set_ylabel(r'$^\circ$C')
#ax.set_title('Hand temperature')

plt.legend()
plt.savefig('fig.pdf',bbox_inches='tight', pad_inches=0)


"""---------------------------------------------------------------------------
---------------------------Correlation----------------------------------------
---------------------------------------------------------------------------"""

for feature in features:
    included = [i if feature==m_total[i][0] else 0 for i in range(len(m_total))]
    chosen = [m_total[i] for i in included if i > 0]
    rho = np.zeros((len(chosen),len(chosen)))
    for i in range(len(chosen)-1):    
        for j in range(i+1,len(chosen)):
            rho[i,j] = np.corrcoef(chosen[i][2],chosen[j][2])[1,0]
    
    print(feature,np.mean(rho[np.nonzero(rho)]))