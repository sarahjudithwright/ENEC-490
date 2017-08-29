import numpy as np
import xlrd
import xlwt


workbook = xlrd.open_workbook('NYGasPrices')




#once upload works
prices = []

for i in range(1,360):
    galdol = workbook(i,1) #becasue start indexing at zero
    prices.append(galdol)
    
    
january = 0
year = 12
annual = []

for i in range(1,30):        
    yravg = np.mean(january:year)
    annual.append(yravg)
    
final = np.zeros[30,2]
final[:,1] == 1987 + arange(30)

final[:,2] == annual


#then figure out how to send back to xls
    