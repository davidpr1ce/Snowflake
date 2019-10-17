#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:09:19 2019

@author: dprice
"""


import numpy as np
import pandas as pd
import progressbar
#making new CSV files where delta flights will have destination
#aerodrome


dbfull = pd.read_csv('~/Documents/CSVnew/full.csv')
allap = dbfull['flight_departure_aerodrome'].dropna().unique()
dbfull=None
del dbfull

files = [ '2019-04-v4.csv', '2019-05-v4.csv', '2019-06-v4.csv', '2019-07-v4.csv', '2019-08-v4.csv']


#TAFI STARTS IN 2019-04.csv
ap = pd.read_csv("~/Documents/CSV/airport-locations-trimmed.csv")
fns = [0,1,2,3,4]


for fn in range(len(fns)):
    d = fns[fn]
    print(files[d])
    db = pd.read_csv("~/Documents/NCSV/" + files[d])
    test = pd.read_csv("~/Documents/NCSV/" + files[d])
    

    #db aerodromes match up with ap airport_codes    
    
    db_DAL = db[db['flight_airline']=='DAL']
    
    mt = db_DAL['metadata_tafi'].dropna() #remove nans in tafi data
    mt = mt.unique()  #find only unique tafis
    
    bar = progressbar.ProgressBar(maxval=len(mt), \
        widgets=[progressbar.Percentage()])
    bar.start()
    
    for i in range(len(mt)): #loop through unique tafis - ie flights from DAL
        #select flight
        flight = db_DAL[(db_DAL['metadata_tafi'] == mt[i])]
        
        #could potentially add an altitude filter here as well, to ensure the plane is low enough...
        
        #final position non rounded
        flat = flight.iloc[-1]['latitude']
        flon = flight.iloc[-1]['longitude']
          
        #find the rounded final lat and lon
        flat_R = np.round(flat/10.)*10.
        flon_R = np.round(flon/10.)*10.
        
        #return rounded arrays of airport locations
        aplat_R = np.round(ap['lat']/10.)*10.
        aplon_R = np.round(ap['lon']/10.)*10.
        
        #find matches 
        i1 = np.where(aplat_R == flat_R)
        i2 = np.where(aplon_R == flon_R)
        
        #find candidates, ie airports within 1 lat and lon of final point
        cand = np.intersect1d(i1,i2) #index of airports in ap database
        cand = cand[np.where(ap['iata_code'][cand].isna() == False)] #remove nans
        
        
        if len(cand) > 0:
            
        
            darr = np.zeros(len(cand))
            for j in range(len(cand)):
                #position of each candidate airport non rounded
                aplat = ap['lat'][cand[j]]
                aplon = ap['lon'][cand[j]]
                #distance between final lat lon and candidate airport
                dist = np.sqrt((aplat - flat)**2. + (aplon - flon)**2.) 
                darr[j] = dist
         
            #find closest airport
            s = np.where(darr == np.min(darr))
        
            #save label of this airport
            airport = ap['airport_code'][cand[s][0]]
        
            #list of indexs in database corresponding to currently selected flight
            indexs = flight.index
        
            #change each destination airport value into database from null to closest
            #airport
            for k in range(len(indexs)):
                test.at[indexs[k],'flight_destination_aerodrome'] = airport
        
        
        bar.update(i+1)
    bar.finish()
        
    
    outname = files[d][0:7] + "-v5.csv"
    test.to_csv("/home/dprice/Documents/NCSV/" + outname)
        
    

