from __future__ import division
import pandas as pd 
import numpy as np


data = pd.read_excel('state_fuel_cf.xlsx',header=0)
data.columns = ['state','fuel','cf']

state = 'NC'
fuel = 'SUN'
cf = .20

# state probabilities
a = data.loc[data['state'] == state]
state_prob = len(a)/len(data)

#fuel probabilities
b = data.loc[data['fuel']==fuel]
fuel_prob = len(b)/len(data)

#capacity factor probability: > 0.2
c = data.loc[data['cf']> cf]
cf_prob = len(c)/len(data)

# conditional probability on selecting NC given SOLAR
# = prob(solar & nc)/prob(solar)

d = a.loc[a['fuel']==fuel]
ncsun = len(d)/len(data)

ncgivensun = ncsun/fuel_prob

# selecting SOLAR given NC

e = b.loc[b['state']==state]
sun_nc = len(e)/len(data)

sungivennc = sun_nc/state_prob

# solar plant in NC with cf > 0.2

f = d.loc[d['cf'] > cf]
ncsolarcf = len(f)/len(data)

# selecting plant with cf > 0.2, conditional on selecting NC solar
# prob(0.2 & ncsolar)/prob(nc&solar)

cfgivenncsun = ncsolarcf/ncsun

# difference between 8.3 and 8.4

diff = cfgivenncsun - ncsolarcf
