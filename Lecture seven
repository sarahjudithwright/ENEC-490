# use .difference function
from __future__ import division
#this defaults division that doesn't give you an integer^^^
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import scipy.stats as stats


coal2012 = pd.read_excel('coal860_data.xlsx',sheetname="2012_coal",header=0)
coal2015 = pd.read_excel('coal860_data.xlsx',sheetname="2015_coal",header=0)
coal2012.columns = ('ID','P_Code','Capacity','Year')
coal2015.columns = ('ID','P_Code','Capacity','Year')


#first create a histogram of 2012 data
years = range(1925,2016)
bins = np.zeros((len(years),1))

for i in range(0,len(coal2012)):
    
    #identifies the year
    yr = coal2012.loc[i,'Year']
    
    #allocates the correct bin
    bin_number = years.index(yr)
    
    #adds capacity to correct bin
    bins[bin_number] = bins[bin_number] + coal2012.loc[i,'Capacity']
    
    
plt.figure()
y_pos = np.arange(len(years))

plt.bar(y_pos,bins, align='center',alpha=0.5,color='0.8',edgecolor='0.5')
tick_range = range(0,len(years),5)
year_ticks = range(1925,2013,5)
plt.xticks(tick_range,year_ticks)
plt.xlabel('Year',fontsize = 10)
plt.ylabel('Capacity',fontsize = 10)

#next add histogram of difference between 
#Identify unique rows that are different

d_2012 = set([ tuple(line) for line in coal2012.values.tolist()])
d_2015 = set([ tuple(line) for line in coal2015.values.tolist()])

# a is a 'set'
a = d_2012.difference(d_2015)

retired = np.array(list(a))

years = range(1925,2016)
bins_retired = np.zeros((len(years),1))

for i in range(0,len(retired)):
    
    #identifies the year
    yr = retired[i,3]
    
    #allocates the correct bin
    bin_number = years.index(yr)
    
    #adds capacity to correct bin
    bins_retired[bin_number] = bins_retired[bin_number] + retired[i,2]
    
bins_2015 = np.zeros((len(years),1))


for i in range(0,len(coal2015)):
    
    #identifies the year
    yr = coal2015.loc[i,'Year']
    
    #allocates the correct bin
    bin_number = years.index(yr)
    
    #adds capacity to correct bin
    bins_2015[bin_number] = bins_2015[bin_number] + coal2015.loc[i,'Capacity']
    


ypos = np.arange(len(years))
plt.figure()
plt.bar(ypos,bins_2015,label='Still Operating')
plt.bar(ypos,bins_retired,label='Retired in 2015')
tick_range = range(0,len(years),5)
year_ticks = range(1925,2016,5)
plt.xticks(tick_range,year_ticks)
plt.xlabel('Year',fontsize = 10)
plt.ylabel('Capacity',fontsize = 10)
plt.title('Existing Coal Capacity by Initial Operating Year')
plt.legend()
