#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:22:21 2019

@author: dprice
"""

#read csv files into workable data blocks

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_all():
    #simple function to read in all the data into one dataframe
    db1 = pd.read_csv("~/Documents/CSV/2018-11.csv")
    db2 = pd.read_csv("~/Documents/CSV/2018-12.csv")
    
    db3 = pd.read_csv("~/Documents/CSV/2019-01.csv")  #(558359, 24)
    db4 = pd.read_csv("~/Documents/CSV/2019-02.csv")
    db5 = pd.read_csv("~/Documents/CSV/2019-03.csv")
    db6 = pd.read_csv("~/Documents/CSV/2019-04.csv")
    db7 = pd.read_csv("~/Documents/CSV/2019-05.csv")
    db8 = pd.read_csv("~/Documents/CSV/2019-06.csv")
    db9 = pd.read_csv("~/Documents/CSV/2019-07.csv")
    db10 = pd.read_csv("~/Documents/CSV/2019-08.csv")

    dbtot = [db1,db2,db3,db4,db5,db6,db7,db8,db9,db10]
    db = pd.concat(dbtot, keys=['a','b','c','d','e','f','g','h','i','j'])
    return db



#one at a time
db = pd.read_csv("~/Documents/CSV/2018-11.csv")


db.dtypes
db['temperature'].describe()


#deleteing columns I don't want for now
dbt = db.drop(["observation_time", "metadata_id", "metadata_origin_time", "metadata_download_time", \
              "metadata_component_name", "flight_airline", "flight_callsign", "flight_registration", \
               "flight_departure_aerodrome", "flight_destination_aerodrome", "latitude", "longitude", \
               "wind_direction", "edr_algorithm", "ncar_mean_confidence", "ncar_peak_confidence", \
               "ncar_number_good_points", "ncar_peak_location", "metadata_tafi"], axis=1)

#dbt = dbt[dbt['edr_peak_value'] > 0.18]
dbt = dbt[dbt['altitude']>0.]

dbz = dbt
#standardising the variables (z score standerdisation)
dbz['altitude'] = ( dbz['altitude'] - dbz['altitude'].mean()) / dbz['altitude'].std() 
dbz['temperature'] = ( dbz['temperature'] - dbz['temperature'].mean()) / dbz['temperature'].std() 
dbz['wind_speed'] = ( dbz['wind_speed'] - dbz['wind_speed'].mean()) / dbz['wind_speed'].std() 
dbz['edr_peak_value'] = ( dbz['edr_peak_value'] - dbz['edr_peak_value'].mean()) / dbz['edr_peak_value'].std() 
dbz['edr_mean_value'] = ( dbz['edr_mean_value'] - dbz['edr_mean_value'].mean()) / dbz['edr_mean_value'].std() 

#calculating the covariance matrix
covm = dbz.cov()
#calculating the correaltion matrix
corm = dbt.corr(method='spearman')

import seaborn as sns


mask = np.zeros_like(corm)
mask[np.triu_indices_from(mask)] = True


#plotting

fig = plt.figure(figsize=(10,10))
#ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

#sns.heatmap(covm, xticklabels=covm.columns, yticklabels=covm.columns, \
#            cmap='bwr', vmin=-1., vmax=1., mask=mask, \
#            annot=True, ax=ax1)

sns.heatmap(corm, xticklabels=corm.columns, yticklabels=corm.columns,  \
            cmap='bwr', vmin=-1., vmax=1., mask=mask, annot=True, ax=ax2)





