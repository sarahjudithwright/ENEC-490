import numpy as np #setting the stage
import xlrd
import xlwt


workbook = xlrd.open_workbook('NYGasPrices') #to upload
#NYGasPrices is the relevent data from the spreadsheet given by the EIA download

prices = []

for i in range(1,360):
    galdol = workbook.cell_value(i,1) #becasue start indexing at zero
    prices.append(galdol)
    
    
january = 0
year = 12
annualavg = []

for i in range(0,30):        
    yravg = np.mean(prices[january:january+year])
    annualavg= np.append(yravg)
    january = january + year
    
final = np.zeros[30,2]
final[:,0] == 1987 + np.arange(30)

final[:,1] == np.transpose(annualavg)

#to export
workbook2 = xlw.Workbook('monthly_average_price.xlsx')
worksheet = workbook2.add_worksheet()

row = 1
col = 0
worksheet.write(col,0,'Year')
worksheet.write(col,1,'Average Price')

for item, price in (final):
    worksheet.write(row,col,item)
    worksheet.write(row,col+1,price)
    row = row + 1

workbook2.close()
    