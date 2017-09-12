# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:01:56 2017

@author: sarahwright
"""

from __future__ import division
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import scipy.stats as stats

df_temps=pd.read_csv('tempdata.csv',header=None)
df_temps.columns = ('Date','Temp')
temps = df_temps.loc[:,'Temp'].as_matrix().astype(np.float)

#Read electricity demand data
df_demand = pd.read_csv('hourly-day-ahead-bid-data-2014.csv',header=4)
# get rid of 'date' column in data
del df_demand['Date']
demand = df_demand.as_matrix().astype(np.float)

#convert to a vector
def mat2vec(data):
    [rows,columns] = np.shape(data)
    vector = []
    for i in range(0,rows):
        vector = np.append(vector,data[i,:])
     
    return vector

vector_demand = mat2vec(demand)

#convert to peak demand vector
peaks = []

for i in range(0,365):
    peak_hourly = np.max(demand[i,:])
    peaks = np.append(peaks,peak_hourly)

peaks = peaks/1000

# forms 2-column matrix
combined = np.column_stack((temps,peaks))

#look for NaNs
for i in range(0,len(combined)):
    if np.isnan(combined[i,1]) > 0:
        combined[i,1] = np.mean([combined[i-1,1],combined[i+1,1]])
        
#clusters for each row
IDX = KMeans(n_clusters=3, random_state=0).fit_predict(combined)

#forms 3-column matrix
clustered_data = np.column_stack((combined,IDX))


plt.figure()
plt.scatter(combined[:,0],combined[:,1],c=IDX+1)
plt.xlabel('Temps (F)',fontsize=24)
plt.ylabel('Electricity Demand (MWh)',fontsize=24)

#
#january average
Jan_Avg = np.zeros((24,1))

for i in range(0,24):
    Jan_Avg[i] = np.mean(demand[0:31,i])
    
plt.figure()
hours = np.arange(1,25)
plt.scatter(hours,Jan_Avg)


#july average    
July_Avg = np.zeros((24,1))

for i in range(0,24):
    July_Avg[i] = np.mean(demand[181:212,i])
    
plt.figure()
hours = np.arange(1,25)
plt.scatter(hours,July_Avg)

'''change nans to zeros - this is okay because averaging 
surrounding values to find a morea ccurate guess would not
 yield a new max'''
demand[np.isnan(demand)] = 0

#column vector
cv = np.zeros((365,1))   
cv[0]=4
for i in range(1,365):
    if cv[i-1] == 7:
        cv[i] = 1
    else:
        cv[i] = cv[i-1] + 1
cv = cv.astype(int)

# max value of each day
maxofday = np.zeros((365,1))
for i in range(0,365):
    maxofday[i] = np.max(demand[i,:])

new = np.concatenate((cv, maxofday),axis=1)

Sun = []
Mon = []
Tues = []
Wed = []
Thurs = []
Fri = []
Sat = []
for i in range(0,365):
    if new[i,0] == 1:
        Sun.append(new[i,1])
    elif new[i,0] == 2:
        Mon.append(new[i,1])
    elif new[i,0] == 3:
        Tues.append(new[i,1])
    elif new[i,0] == 4:
        Wed.append(new[i,1])
    elif new[i,0] == 5:
        Thurs.append(new[i,1])
    elif new[i,0] == 6:
        Fri.append(new[i,1])
    elif new[i,0] == 7:
        Sat.append(new[i,1])

box = [Sun,Mon,Tues,Wed,Thurs,Fri,Sat]
 
#plot of max of each day by day of week       
plt.figure()
plt.boxplot(box)

'''we notice that weekends are lowest and 
it generally declines throughout the week'''
