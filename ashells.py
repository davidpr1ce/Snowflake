"""
=======================
Pie chart on polar axis
=======================

Demo of bar plot on a polar axis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



latlongrid = [-140, 15, -95, 65]    #US
latlongrid2 = [-140, 15, -65, 65] 
#latlongrid = [-30, 25, 55, 70]  #EU



#shells = np.linspace(5000, 15000, num=11)
shells = np.linspace(29000, 41000, num=13)
db = get_db('2019-08-01 00:00:00+00', '2019-08-30 00:00:00+00', latlongrid=latlongrid)

db = db[db['edr_peak_value']>0.5]
db = db[db['altitude']>30000]

tafis = db['metadata_tafi'].unique()



flight = flight_info('08', tafis[0], Plot=True, APlot=True, Afilter=[30000,40000])



#pchance, mchance = alt_shells(db, latlongrid2, shells, edr_peak=0.15,vmax=15., Plot=True)





#HISTOGRAM PLOTTING - should probably add as an option to alt_shells
'''
files = ["2019-04-v2.csv","2019-05-v2.csv","2019-06-v2.csv","2019-07-v2.csv","2019-08-v2.csv"]

colour= ['cyan', 'orange', 'pink', 'purple', 'green', 'blue', 'red', 'thistle', 'lime', 'slategrey']

pcs = np.zeros((len(files), len(shells)-1))

for i in range(len(files)):
    db = pd.read_csv("~/Documents/CSVnew/" + files[i])
    pchance, mchance = alt_shells(db, latlongrid, shells, Plot=True)
    pchance = (pchance/np.sum(pchance)) * mchance   #linear scaling of percentages to make sum over all shells - maths checks out..
     
#    print(np.sum(pchance), mchance)
    pcs[i,:] = pchance

fig, axs = plt.subplots(1,1,figsize=(10,10)) 
cmp = np.zeros(len(files))
for i in range(len(pcs[0,:])):
    axs.bar(range(len(files)), pcs[:,i], width=1.0, color=colour[i], edgecolor='black', align='center', bottom=cmp )
    cmp = cmp + pcs[:,i]
    axs.set_xticklabels([' ', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08'])
    axs.set_ylabel('Percentage Chance of Turbulent event')
    axs.set_ylim(0,20)
    axs.legend([(str(int(shells[i])/1000) + ' - ' + str(int(shells[i+1])/1000) + ' kft') for i in range(len(shells)-1) ])

'''

