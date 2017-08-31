# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:17:45 2017

@author: sarahwright
"""

import xlrd
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

workbook = xlrd.open_workbook('monthly_demandNC.xls')
sheet = workbook.sheet_by_index(0)

demand = []

for i in range(0,216):
    value = sheet.cell_value(i,0)
    demand.append(value)
    


#from lecture three
def annual_profile(demand):
    
    #number of years in record
    num_years = int(len(demand)/12)
    
    #output matrix of zeros
    output = np.zeros((12,num_years))
    
    #nested for loops
    for i in range(0,num_years):
        for j in range(0,12):
            output[j,i] = demand[(i*12)+j]
    return output 
 
m = annual_profile(demand)

#for year on year, (current/previous)/previous

'''def diff(m):
    differences = np.zeros(12, num_year)

    for i in range(0,12):
        for j in range(0,num_years):
            differences(j,i) = (m(j)-m(j+1))/m(j)'''
