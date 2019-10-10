#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:15:54 2019

@author: dprice
"""

#regions of interest
NA = [-80,25,10,65]
US = [-140,15,-50, 65]
JP = [80,15,180,65]
EU = [-30,25,55,70]
PC = [110, 5, -110, 75]
GLOBE = [-180, -70, 180, 70]
ATLAS = [-10, 25, 10, 40]


    
def read_all(*, New=False):
    import pandas as pd
    #simple function to read in all the data into one dataframe
    
    if (New==True):
        db1 = pd.read_csv("~/Documents/CSVnew/2019-04-v3.csv")
        db2 = pd.read_csv("~/Documents/CSVnew/2019-05-v3.csv")
        db3 = pd.read_csv("~/Documents/CSVnew/2019-06-v3.csv")
        db4 = pd.read_csv("~/Documents/CSVnew/2019-07-v3.csv")
        db5 = pd.read_csv("~/Documents/CSVnew/2019-08-v3.csv")
        
        dbtot=[db1,db2,db3,db4,db5]
        db = pd.concat(dbtot, keys=['a', 'b', 'c', 'd', 'e'])
        return db
    else:
    
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


def get_PArr(filename,*, altitude=[0,50000], binsize=4., edr_peak=0.18, \
             region=[-180,-65,180,75], pass_db=False):

    #llcrnrlon=-180.,llcrnrlat=-65,urcrnrlon=180.,urcrnrlat=75
    '''
    Function to produce a interpolated 2darray in lat and lon axis
    of % chance of encountering a turbulent event within given paraneters.
    Inputs:
        filename - CSV file of data you want to read, eg, "2019-01.CSV"
        altitude - altitude you want to use as cutoff. eg. 0 will be all points, 20000 is all points above this
        binsize - binsize for 2dhistograms - effects the resolution of the contour
        edr_peak - EDR threshold that you want to flag as a turbulent event
                    This is a complex threshold, leave at 0.18 for now... 
        region = [x1, y1, x2, y2] - lat lon region to filter data into, IDL position syntax
        pass_db - set to True to pass a database directly to function in filename position instead of reading one in
    '''
    import pandas as pd
    import numpy as np
    from scipy import interpolate     

    #reading and filtering data
    if (pass_db == True):
        dbt = filename
    else:
        dbt=pd.read_csv("~/Documents/CSV/" + filename)
    
    dbt=dbt[(dbt['altitude']>altitude[0]) & (dbt['altitude']<altitude[1])]
    
    #filtering lat lon region
    dbt=dbt[(dbt['longitude']>region[0]) & (dbt['longitude']<region[2]) \
            & (dbt['latitude']>region[1]) & (dbt['latitude']<region[3])]
    
    events=dbt[dbt['edr_peak_value']>edr_peak]    
    
    if len(events)<10: events = pd.DataFrame(columns=dbt.columns) #handling a fringe case where there is one event which breaks histogram2d...

    
    binn=[360/binsize,180/binsize]      
    #binn=[(region[2]-region[0])/binsize, (region[3]-region[1])/binsize]   
    Hall, xedges, yedges = np.histogram2d(dbt['longitude'], dbt['latitude'], bins=binn, \
                                          range=[[-180,180],[-90,90]])    
    Hall[np.where(Hall < 3*np.mean(Hall))] = 0.    
    Heve, xedges, yedges = np.histogram2d(events['longitude'], events['latitude'], \
                                          bins=binn, range=[[-180,180],[-90,90]])    
    
    Heve[np.where(Hall < 10*np.mean(Hall))] = 0.    
    Perc = ((Heve/Hall)*100.)
    #interpolating
    X = np.arange(binn[0])*binsize - 180. + (binsize/2.)
    Y = (np.arange(binn[1])*binsize - 90. + (binsize/2.))*-1.    
    Perc[np.isnan(Perc)] = 0. 
    Hall[np.isnan(Hall)] = 0.
    f = interpolate.interp2d(X,Y,np.transpose(Perc),kind='linear') #cubic gives negative values
    g = interpolate.interp2d(X,Y,np.transpose(Hall),kind='linear')
    X=np.arange(360)-180 + .5
    Y=(np.arange(180)-90 +.5)*1.  #for some reason interpolating flips the latitude so this is *1 now...
    Perc = f(X,Y)
    Hall = g(X,Y)
    
    #removing interpolation artifacts
    Perc[np.where(Perc < np.nanmax(Perc)*0.01)] = np.nan
    Hall[np.where(Hall < np.nanmax(Perc)*0.01)] = np.nan
    return X, Y, Perc, Hall


    
def plot_mat_BM(H, ax,*, cmap='Blues', title=' ', ctitle=' ', vmin=0, vmax=10, \
                nlevels=10, alpha=0.75, region=[-180,-65,180,75]):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.basemap import Basemap
    
    X=np.arange(360)-180 + .5
    Y=(np.arange(180)-90 +.5)*-1.
    
    m = Basemap(resolution='c', epsg=3857, ax=ax, \
                llcrnrlon=region[0],llcrnrlat=region[1], urcrnrlon=region[2], \
                urcrnrlat=region[3])
    m.drawcoastlines()
    m.fillcontinents(alpha=0.2, color='green')
    m.drawcountries()
    #m.drawstates()
    
    X=np.arange(360)-180 + .5
    Y=(np.arange(180)-90 +.5)*-1. 
    XY=np.meshgrid(X,Y)    
    #levels = np.linspace(vmin,vmax,nlevels)
   #mplot = m.contourf(XY[0],XY[1],H, levels=levels, alpha=alpha, cmap=cmap, latlon=True)
    mplot = m.pcolormesh(XY[0], XY[1], H, cmap=cmap, latlon=True, vmin=vmin, vmax=vmax)
    cb=plt.colorbar(mplot, ax=ax, cmap=cmap, fraction=0.02, pad=0.04)
    cb.set_label(ctitle)
    ax.set_title(title)
    
    
def full_corr(edr_peak, altitude):
    '''
    Function to produce the correlation matrix for the full dataset
    Inputs:
        edr_peak - EDR threshold, same caveats apply
        altitude - altitude threshold (data taken above)
    '''
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    dbt = pd.read_csv("~/Documents/CSV/full_trimmed.csv")

    dbt = dbt.drop(["observation_time", "flight_departure_aerodrome", "flight_destination_aerodrome", "latitude",  \
                "longitude", "wind_direction", "ncar_mean_confidence", "ncar_peak_confidence", "metadata_tafi"], axis=1)

    mask = np.zeros((5,5), dtype=bool)
    mask[0,1:5] = True
    mask[1,2:5] = True
    mask[2,3:5] = True
    mask[3,4] = True
    #fig = plt.figure(figsize=(10,10))
    dbt = dbt[dbt['edr_peak_value'] > edr_peak]
    dbt = dbt[dbt['altitude'] > altitude]
    dbt = dbt.drop(['Unnamed: 0','Unnamed: 1'], axis=1)
    corm = dbt.corr(method='spearman')
    ax= plt.subplot(111)
    labels = ['Altitude', 'Temperature', 'Wind Speed', 'EDR Peak Value', 'EDR Mean Value']
    sns.heatmap(corm, xticklabels=labels, yticklabels=labels,  \
            cmap='bwr', vmin=-1., vmax=1., mask=mask, annot=True, ax=ax)
    
    

    
def get_AP_name(code, *, back=False):
    '''
    Simple function to print airport name from its code
    If optional keyword back is set to True does the reverse operation.
    '''
    import pandas as pd
    import numpy as np
    ap = pd.read_csv("~/Documents/CSV/airport-locations.csv")
    
    if back==True:
        name = ap['airport_code'][np.where(ap['airport_name']== code)[0][0]]
    else:
        name = ap['airport_name'][np.where(ap['airport_code'] == code)[0][0]]
    
    return name
    
 
def get_flight(month, tafi):
    '''
    Simple function to return a pandas database of an individual flight, indentified by
    the month and by its metadata_tafi tag.
    Possibly upgrade to entire dataset without month keyword..
    
    Inputs:
        month (string) - month of flight (2019-04 onwards) eg. "2019-05"
        tafi (string) - metadata tafi string of flight
    '''
    import pandas as pd
    print('get_flight is currently using v5 CSV files (w/ report flags and new DAL destinations.)')
    db = pd.read_csv("~/Documents/CSVnew/2019-" + month + "-v5.csv")
    flight = db[db['metadata_tafi'] == tafi]
    
    
    return flight
    
def flight_info(flight, *, Plot=True, MaskT=False, \
                Afilter=[0,50000], EDRf = [0,0], Tempfix=True, APlot=False, \
                Velocity=0.):
    '''
    Return and/or plot desired information for a given flight tafi in a specific month
    Inputs:
        Passed to get_flight:
            flight - pandas dataframe of the flight in question
        Keywords:
            GCircle - set to True to produce great circle plot
            Panels - set to True to produce panels plot
            Aplot - set to True to plot altitude plot
            Afilter - set to filter the altitude regions between Afilter[0] and Afilter[1]
            EDRf - set to filter mean EDRf[0] and peak EDRf[1] turbuluence reports
            
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd  
    #keyword handling - for plotting later
    npanels = 5
    if (APlot == True): npanels = 6


    
    #altitude independent temperature attempt
    if (Tempfix==True):
        #fit a polynomial to atl-temp points
        p, V = np.polyfit(flight['altitude'], flight['temperature'], 2, cov=True)
        def calc_T(A):
            '''return temp associated with an alt A'''
            return p[0]*A**2 + p[1]*A + p[2]
           
    #filter into altitude range
    flight = flight[(flight['altitude'] > Afilter[0]) & (flight['altitude'] < Afilter[1])]
    #filter EDR values
    flight = flight[(flight['edr_mean_value'] >= EDRf[0]) & (flight['edr_peak_value'] >= EDRf[1])]
    #produce a warning if you filter all points out of flight
    if (len(flight) == 0):
        print('###############################################################')
        print('### WARNING: Filtered all points! Relax keyword conditions. ###')
        print('###############################################################')
        return flight
    
    
    time = flight['observation_time'] 
    ntime =np.array(time)

    #converting to seconds for plotting
    time = np.array([sum(x * int(t) for x, t in zip([3600, 60, 1], time.iloc[i][11:19].split(":"))) \
                for i in range(len(time))])
        
    #dealing with flights that traverse midnight
    if ((np.all(time[-1] <= time[1])) == False):  #WAS time[:-1] <= time[1:] changed 1st oct
        time[np.where(time < time[0])] = time[np.where(time < time[0])] + 86400

    #masking high turbulent events.. by filtering out non-heartbeat length datapoints
    if (MaskT == True):
        
        dt = np.ediff1d(flight['utc_timestamp']) #array of time diff
        dt = pd.Series(np.insert(dt, 0, np.nanmax(dt)), index=flight['utc_timestamp'].index) #series of time diff( with max inserted at start to match lengths)
        mask = (dt != np.nanmax(dt))  #mask of values where dt is less than heartbeat
        nani = mask[mask==True].index
        flight.loc[nani] = np.nan
      
    #add a plot of adjusted temperature gradient (later just handling atm)
    if (Tempfix==True):
        temp = np.array(flight['temperature'])
        alt = np.array(flight['altitude'])
        #adjusted temperature for calculating non altitude affected gradient!
        temp_a = temp - calc_T(alt)
        #gradient of new temperature time series
        tempg = np.gradient(temp_a, time)

 
    #get departure and arrival airports
    depA, arrA = flight['flight_departure_aerodrome'].iloc[0], flight['flight_destination_aerodrome'].iloc[0]
    
    #average wind direction and velocity calculations
    #going to calculate heading by average of gradients 
    import metpy.calc as mp
    
    heading = mp.wind_direction(np.gradient(flight['longitude'],time), np.gradient(flight['latitude'],time)).magnitude
    Ev, Nv = mp.wind_components(np.array(flight['wind_speed']),np.radians(np.array(flight['wind_direction']))) 
    windd = mp.wind_direction(Ev, Nv).magnitude 
    
    #minimum difference in angle (non-directional) to avoid wrapping problems 0-360
    angd =  180- np.abs(np.abs(windd-heading) -180)
    
    wind_component_Series = pd.Series(flight['wind_speed']*np.cos(np.radians(angd)))
    
    avwd = np.nanmean(angd)
    avws = flight['wind_speed'].mean()
    component_wind = avws* np.cos(np.radians(avwd))




    #both plots!!!
    if (Plot==True):
        fig, axs = plt.subplots(1,1, figsize=(24,12))
        plt.subplots_adjust(left=0.45, hspace=0.4)
        from mpl_toolkits.basemap import Basemap
        import metpy.calc as mp
        #find lat and lon of initial and final airports
        #[ilat, ilon, flat, flon]
    
        pnts = [np.min(flight['latitude']), np.min(flight['longitude']), np.max(flight['latitude']), np.max(flight['longitude'])]
        latscl = np.sqrt((np.max(flight['latitude']) - np.min(flight['latitude']))**2.)
        lonscl = np.sqrt((np.max(flight['longitude']) - np.min(flight['longitude']))**2.)
        
        corners=[pnts[1]-0.3*lonscl, pnts[0]-latscl,pnts[3]+0.3*lonscl,pnts[2]+latscl]
        
        #handling edge cases
        if corners[0] < -180: corners[0] = -180
        if corners[1] < -75: corners[1] = -75
        if corners[2] > 180: corners[2] = 180
        if corners[3] > 75: corners[3] = 75
        

        bm = Basemap(resolution='l', epsg=3857, ax=axs, \
                     llcrnrlon=corners[0] ,\
                     llcrnrlat=corners[1] ,\
                     urcrnrlon=corners[2] ,\
                     urcrnrlat=corners[3] )
        bm.drawcoastlines()
        bm.fillcontinents(alpha=0.3, color='green')
        bm.drawcountries()
        #bm.drawgreatcircle(pnts[1],pnts[0],pnts[3],pnts[2], color='black')
        
        Ev, Nv = mp.wind_components(np.array(flight['wind_speed']),np.radians(np.array(flight['wind_direction'])))
        
        sp = bm.scatter(np.array(flight['longitude']), np.array(flight['latitude']), latlon=True, \
                   c=list(flight['edr_mean_value']), marker='o', cmap='viridis', vmin=EDRf[0],\
                   vmax=np.max(flight['edr_mean_value']), zorder=2)
        
        #adding surrounding points
        
        
        
        qp = bm.quiver(np.array(flight['longitude']), np.array(flight['latitude']), Ev, Nv, latlon=True, \
                 width=0.003, pivot='tail', color='blue', zorder=1)

        #plotting an extra quiver plot with the vector components on

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ia = inset_axes(axs, width="10%", height="20%", loc='upper left')
        ia.get_xaxis().set_visible(False)
        ia.get_yaxis().set_visible(False)        

        #need to normalise these vectors to solve scaling issues i think      
        Heading_vec = [np.mean(np.gradient(flight['longitude'],time)), np.mean(np.gradient(flight['latitude'],time)) ]
        Heading_vec = Heading_vec/np.linalg.norm(Heading_vec)
        Wind_vec = [np.mean(Ev),np.mean(Nv) ]
        Wind_vec = Wind_vec/np.linalg.norm(Wind_vec)
        
        #plotting       
        ia.quiver(0, 0, Heading_vec[0], Heading_vec[1], width=0.03, scale=2.)
        ia.quiver(0,0, Wind_vec[0], Wind_vec[1], width=0.03, color='blue', scale=2. )
        #annotating        
        if (avwd < 90.):
            hort = 'Tail wind'
        else:
            hort = 'Head wind'            
        ia.annotate(hort, [0.22,0.83], xycoords='axes fraction')
        ia.annotate(str(np.round(component_wind, decimals=1)) + ' Knots', [0.2, 0.05], \
                    xycoords='axes fraction')
        
        
        #title = get_AP_name(depA) + ' - ' + get_AP_name(arrA) + ' : ' + 'Turbulence, wind velocity and direction' 
        title= ' '
        axs.set_title(title)
        axs.set_xlabel('Average wind velocity w.r.t aircraft heading: ' \
                       + str(np.round(component_wind, decimals=1)) + ' Knots')
        key_mag = int(np.round(flight['wind_speed'].mean()/100., decimals=1)*100.) #round to closest 10
        plt.quiverkey(qp, 0.1, -0.025, key_mag, str(key_mag) +' Knots', fontproperties={'weight': 'bold'}, labelpos='W')
        cb = plt.colorbar(sp, fraction=0.02, pad=0.04, cmap='viridis')
        cb.set_label('Mean EDR Value')
        
        #PANELS
        fig2, axs2 = plt.subplots(npanels,1, figsize=(12,12), sharex=True)
        plt.subplots_adjust(bottom=0.15, hspace=0.25)
        plt.xticks(rotation=60)
                
        #xticks and labels handeling..     
        dt = time[-1]-time[0]
        step = np.max(np.ediff1d(time))
        nsteps = dt/step     
        time = time-time[0]
        xticks = [time[0] + i*step for i in range(int(nsteps)+1)]
        
        #Janky AF xlabels.... try and decipher that lmao. time[i] is seconds from first time (float), ntime[0] is a string of the first time, 
        #so convert that to float, add time[i] and then convert the sum back to a string and set as the xlabel... 
        xlabels = [(time_transform(xticks[i] + time_transform(ntime[0]),Back=True))[10:] for i in range(len(xticks))]
        
        #ornot
        #xticks=time        

            
        axs2[0].plot(time, flight['edr_mean_value'], color='black', marker='o', linestyle='--')
        axs2[0].plot(time, flight['edr_peak_value'], color='black')
        axs2[0].legend(['EDR Mean', 'EDR Peak'])
        axs2[0].set_ylabel('EDR')
        axs2[0].set_title('Reported EDR')
        
        axs2[1].plot(time, flight['temperature'], color='r', marker='o')
        axs2[1].set_ylabel('(K)')
        axs2[1].set_title('Temperature')
    

        
        axs2[2].plot(time, tempg, color='r', marker='o' )
        axs2[2].plot(time, np.gradient(flight['temperature'],time), color='r', linestyle='--',marker='^')
        axs2[2].legend(['Adj. Temp. gradient','Temp. Gradient'])
        axs2[2].set_ylabel('(K/s)')
        axs2[2].set_title('Temperature Gradient')


        axs2[3].plot(time,wind_component_Series, color='blue', marker='o')
        axs2[3].set_ylabel('(Knots)')
        axs2[3].set_title('Wind component velocity with respect to aircraft heading')

        axs2[4].plot(time,np.gradient(wind_component_Series,time), color='blue', linestyle='--',marker='o')
        axs2[4].set_ylabel('(Knots/s)')
        axs2[4].set_title('Wind component gradient')
      
        if (npanels > 5):
            axs2[5].plot(time, flight['altitude'], color='g', marker='o')
            axs2[5].set_ylabel('(ft)')
            axs2[5].set_xticks(xticks)
            axs2[5].set_xlabel(ntime[0][0:10])
            axs2[5].set_title('Altitude')
            axs2[5].set_xticklabels(xlabels)
        else:
            axs2[4].set_xticks(xticks)
            axs2[4].set_xlabel(ntime[0][0:10])
            axs2[4].set_xticklabels(xlabels)

        
    #adding interesting data columns
    flight['adjusted_temp_gradient'] = pd.Series(tempg).values
    flight['wind_component_gradient'] = pd.Series(np.gradient(wind_component_Series,time)).values
    flight['seconds_since'] = pd.Series(time).values
    
    #very simple calculation of difference in flight time due to wind.
    if (Velocity > 0.):
        from haversine import haversine
        ap = pd.read_csv("~/Documents/CSV/airport-locations.csv")
        depA, arrA
        
        #in km
        distance = haversine([ ap[ap['airport_code']==depA]['lat'].iloc[0], ap[ap['airport_code']==depA]['lon'].iloc[0]], \
                               [ap[ap['airport_code']==arrA]['lat'].iloc[0], ap[ap['airport_code']==arrA]['lon'].iloc[0]])
        
        time_nowind = (distance*1000)/Velocity
        time_wind = (distance*1000)/(Velocity+component_wind)
        
        time_diff = time_wind - time_nowind
        
        print('Time difference due to wind: ' + str(time_diff/60.) + ' mins')
    return flight
 



def find_flights(db, departure_ap, destination_ap, *, Both=True):
    ''' 
    Function to search for every flight between two airports in the dataset.
    Returns an array of tafi tags for each flight.
    Inputs:
        db - database to search
        departure_ap (string)... - departure and destination aerodromes (codes)
        both - return both directions or just one - default is True (both directions)
    '''
    import pandas as pd
        
    #filter results
    db1 = db[(db['flight_departure_aerodrome'] == departure_ap) & (db['flight_destination_aerodrome'] == destination_ap)]
    
    if (Both==True):
        db2 = db[(db['flight_departure_aerodrome'] == destination_ap) & (db['flight_destination_aerodrome'] == departure_ap)]
        db = pd.concat([db1,db2])
    else:
        db= db1
    
    #get tafis    
    #tafis = db['metadata_tafi'].dropna().unique()
#    month = mt['observation_time'].unique()
#    month = [month[i][5:7] for i in range(len(month))]
   #db = db[['flight_departure_aerodrome','flight_destination_aerodrome','metadata_tafi']]
    
    return db #tafis
    

def alt_shells(db, latlongrid, shells, *, Plot=True, edr_peak=0.18, vmax=50.):
    '''
    Investigate frequency of turbulence by altitude
    Input:
        db - pandas dataframe of selected time of interest..
        latlongrid - [minlon, minlat, maxlon, maxlat] boundaries of spatial region to investigate.
        shells - array of alitude shells to investigate (any length) eg. [20000, 22000, 24000, 26000 etc...]
        
    Returns an array of % chance to encounter turbulence at each shell
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    
    db = db[ (db['latitude'] > latlongrid[1]) & (db['latitude'] < latlongrid[3]) & \
             (db['longitude'] > latlongrid[0]) & (db['longitude'] < latlongrid[2]) ]
    
    print("No. of data points: " + str(len(db)))    
    
    if (len(db)==0):
        print("No data points!!!")
        return
    
    nplots = len(shells) -1 
    
    if (Plot == True):
        fig, axs = plt.subplots(int(nplots/2),2, figsize=(nplots*1.2,nplots*2))
        axs = axs.flatten()
    
    
    #this month over whole shell range
    Xm, Ym, Hm, Hallm = get_PArr(db, altitude=[shells[0],shells[-1]], \
                                 region=[latlongrid[0],latlongrid[1], latlongrid[2], latlongrid[3]], \
                                 pass_db=True, edr_peak=edr_peak)
    mchance = np.nanmean(Hm)
    pchance = np.zeros(nplots)

    for i in range(nplots):
        #get percentage occurence 2d array
        X, Y, H, Hall  = get_PArr(db, altitude=[shells[i], shells[i+1]], \
                                  region=[latlongrid[0],latlongrid[1], latlongrid[2], latlongrid[3]], \
                                  pass_db=True, edr_peak=edr_peak)
        
        tit = 'Altitude Shell: ' + str(shells[i]) + ' - ' + str(shells[i+1]) + ' ft.'
        
        if (Plot == True):
        
            plot_mat_BM(H, axs[i], region=[latlongrid[0],latlongrid[1], latlongrid[2], latlongrid[3]], 
                        vmin=0, vmax=vmax, title=tit, ctitle='Percentage Chance')
            axs[i].set_xlabel('Mean/Max Percentage chance: ' + str(np.round(np.nanmean(H),decimals=1)) + ' / ' + str(np.round(np.nanmax(H), decimals=1)))
        
        pchance[i] = np.nanmean(H)
    
 
    #return percentage in each shell, and total percentage chance for the month please
    return pchance, mchance

        
def get_db(start,end, *, latlongrid=[-80, 25, 10, 65], Alayer=[0,50000]):
    '''
    function to return dataframe of events within a time window start - end
    Inputs:
        start - start time eg. 2019-04-01 12:00:00+00
        end - end time (same format)
        latlongrid - grid you want to plot, defaults to north atlantic
    Idea is for this to facilitate the past_conditions function below
    '''
    import pandas as pd
    import datetime
    import numpy as np

    #determine what month db file to read, rather than whole thing - for speed    
    imonth = start[5:7]
    fmonth = end[5:7]
    
    print('get_db currently set to use v5 files (w/ event flags and new DAL destinations)')
    #months are the same:
    if (int(imonth)==int(fmonth)):
    
        fn = '~/Documents/CSVnew/2019-' + imonth + '-v5.csv'
        db = pd.read_csv(fn)
      
        start_ts = datetime.datetime.strptime(start+'00','%Y-%m-%d %H:%M:%S%z').timestamp()
        end_ts = datetime.datetime.strptime(end+'00','%Y-%m-%d %H:%M:%S%z').timestamp()      
        #array of timestamps for all observation times
        timestamps = np.array(db['utc_timestamp'])
    
        #old way with old files w/o utc_timestamp column
        #timestamps = np.array([datetime.datetime.strptime(obtimes[i][0:19] + '+0000', '%Y-%m-%d %H:%M:%S%z').timestamp() for i in range(len(obtimes))])
        
        #might work...
        index = np.where(timestamps < end_ts)
        index = np.where(timestamps[index] > start_ts)[0] 
        db = db.iloc[index,:]
        #did work! haha
        
        db = db[ (db['latitude'] > latlongrid[1]) & (db['latitude'] < latlongrid[3]) & \
                 (db['longitude'] > latlongrid[0]) & (db['longitude'] < latlongrid[2]) ]
        
        db=db[(db['altitude']>Alayer[0]) & (db['altitude']<Alayer[1])]
        return db
    #time period straddles a single month boundary
    if (int(fmonth) - int(imonth)) == 1:
        fn1 = '~/Documents/CSVnew/2019-' + imonth + '-v5.csv'
        fn2 = '~/Documents/CSVnew/2019-' + fmonth + '-v5.csv'
        
        db1 = pd.read_csv(fn1)
        ts1 = np.array(db1['utc_timestamp'])
        db2 = pd.read_csv(fn2)
        ts2 = np.array(db2['utc_timestamp'])
        
        start_ts = datetime.datetime.strptime(start+'00','%Y-%m-%d %H:%M:%S%z').timestamp()
        end_ts = datetime.datetime.strptime(end+'00','%Y-%m-%d %H:%M:%S%z').timestamp()         
        
        i1 = np.where(ts1 > start_ts)[0]
        db1 = db1.iloc[i1,:]
        
        i2 = np.where(ts2 < end_ts)[0]
        db2 = db2.iloc[i2,:]
        
        db = pd.concat([db1,db2])
        
        db = db[ (db['latitude'] > latlongrid[1]) & (db['latitude'] < latlongrid[3]) & \
                 (db['longitude'] > latlongrid[0]) & (db['longitude'] < latlongrid[2]) ]
        
        db=db[(db['altitude']>Alayer[0]) & (db['altitude']<Alayer[1])]
        return db
    #time period straddles multiple months
    if (int(fmonth) - int(imonth)) > 1:
        fn1 = '~/Documents/CSVnew/2019-' + imonth + '-v5.csv'
        fn2 = '~/Documents/CSVnew/2019-' + fmonth + '-v5.csv'
        
        db1 = pd.read_csv(fn1)
        ts1 = np.array(db1['utc_timestamp'])
        db2 = pd.read_csv(fn2)
        ts2 = np.array(db2['utc_timestamp'])
        
        start_ts = datetime.datetime.strptime(start+'00','%Y-%m-%d %H:%M:%S%z').timestamp()
        end_ts = datetime.datetime.strptime(end+'00','%Y-%m-%d %H:%M:%S%z').timestamp()         
        
        i1 = np.where(ts1 > start_ts)[0]
        db1 = db1.iloc[i1,:]
        
        i2 = np.where(ts2 < end_ts)[0]
        db2 = db2.iloc[i2,:]

        #find number of intermediate months
        n_intermediate_db = (int(fmonth) - int(imonth)) - 1

        #loop through int. months and save them into a list for concat later
        db_list = []
        for i in range(n_intermediate_db):
            month = i+int(imonth)+1
            db_i = pd.read_csv('~/Documents/CSVnew/2019-' + str(month).zfill(2) + '-v5.csv')
            db_list.append(db_i)
    
        #concat all the dataframes together
        db = pd.concat([db1, pd.concat(db_list), db2])
        return db


def past_conditions(ctime, window, *, latlongrid=[-80,25,10,65], Alayers=[0,50000], \
                    resolution=2., contour=None, Grid=False, Gradients=None):
    '''
    function to produce a plot of conditions in the time window prior to the given 'current' time
    Inputs:
        ctime - time string of end time/current time eg. '2019-08-01 12:00:00+00'
        window - time window you want to look back over, in hours?mins?
        latlongrid - region you want to investigate - defaults to north atlantic
        Alayers - altitude layers you want to seperate the data out into
        Resolution - resolution of wind gridding - use with caution!!
    output:
        plot of conditions in the latlongrid over that window
        maybe some recommendations we wil see
    '''
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import metpy.calc as mp
    
    utc_ctime = time_transform(ctime)
    window_s = window * 60. * 60
    utc_etime =utc_ctime - window_s
    start = time_transform(utc_etime, Back=True) 

    nlayers = len(Alayers)-1
    nrows=1
    fig, axs = plt.subplots(nlayers,nrows,figsize=(12*nlayers,32*nrows))
    
    #some axis handling for the following loop
    if (nlayers + nrows > 2): axs=axs.flatten()
    if (nlayers + nrows ==2): axs = [axs]
    
    
    db = get_db(start, ctime, latlongrid=latlongrid, Alayer=[Alayers[0], Alayers[-1]])
    
    if (Gradients != None):
        db=gradients(start, ctime, key=Gradients, latlongrid=latlongrid, Alayers=Alayers, resolution=resolution, grid=True)
        return db
    else:
    
        for i in range(nlayers):
            
            dbf = db
            dbf = dbf[(dbf['altitude'] >= Alayers[i]) & (dbf['altitude'] <= Alayers[i+1])]        
    
            
            #add loop here over some key layers eg. for nat tracks
        
    #        bm = Basemap(resolution='l', epsg=3857, ax=axs[i], \
    #                     llcrnrlon=latlongrid[0] - (latlongrid[2]-latlongrid[0])*0.2 ,\
    #                     llcrnrlat=latlongrid[1] - (latlongrid[3]-latlongrid[1])*0.1,\
    #                     urcrnrlon=latlongrid[2] + (latlongrid[2]-latlongrid[2])*0.2,\
    #                     urcrnrlat=latlongrid[3] + (latlongrid[3]-latlongrid[1])*0.1)
    
            bm = Basemap(resolution='l', epsg=3857, ax=axs[i], \
                         llcrnrlon=latlongrid[0] ,\
                         llcrnrlat=latlongrid[1] ,\
                         urcrnrlon=latlongrid[2] ,\
                         urcrnrlat=latlongrid[3] )
            
            bm.drawcoastlines()
            bm.fillcontinents(alpha=0.4, color='green')
            bm.drawcountries()
            if (Grid==True):
                para = np.arange(0.,81.,10.)
                meri = np.arange(10., 351., 20.)
                bm.drawparallels(para, labels=[True])
                bm.drawmeridians(meri, labels=[True])
                
            #grid up edr vectors?
            if (Grid==True):        
                dbedrm = grid_db(dbf, 'edr_mean_value', latlongrid, resolution=resolution)
                sp = bm.scatter(np.array(dbedrm['longitude']), np.array(dbedrm['latitude']), latlon=True, \
                                c=list(dbedrm['edr_mean_value']),s=list(dbedrm['edr_mean_value']*250), marker='o', cmap='viridis', vmin=0, \
                                vmax=np.max(dbedrm['edr_mean_value']), zorder=2)
            else:
            #or dont
    
                sp = bm.scatter(np.array(dbf['longitude']), np.array(dbf['latitude']), latlon=True, c=list(dbf['edr_mean_value']), \
                                s=list(dbf['edr_mean_value']*250), marker='o', cmap='viridis', vmin=0, \
                                vmax=0.2, zorder=2)        
    
            
            #trying to add contour plot
            xc, yc, dbcont = create2d_data(dbf, 'edr_mean_value', latlongrid, resolution=resolution)
            
            #dbcont[np.where(dbcont<0.01)] = 0
            
            if (Grid == True):
                xx, yy = np.meshgrid(xc,yc)
                cp = bm.pcolormesh(yy, xx, dbcont, cmap='Reds', zorder=0, latlon=True, alpha=0.5)
            else:
                xx, yy = np.meshgrid(xc+resolution/2.,yc+resolution/2.) #pcolormesh and contourf have different coord requirments...
                cp = bm.contourf(yy, xx, dbcont, cmap='Reds', levels=250, zorder=0, latlon=True)
    
            
            #grid up wind vectors to avoid oversampling in some regions
            
            if (Grid == True):
                dbwind = grid_db(dbf, 'wind_vector', latlongrid, resolution=resolution)
                EWv = np.array(dbwind['EW_velocity'])
                NSv = np.array(dbwind['NS_velocity'])
                
            else:
                dbwind = dbf
                EWv, NSv = mp.wind_components(np.array(dbwind['wind_speed']), np.radians(np.array(dbwind['wind_direction'])))
    
            qp = bm.quiver(np.array(dbwind['longitude']), np.array(dbwind['latitude']),\
                           EWv,NSv , latlon=True, \
                           width=0.003, pivot='tail', color='blue', zorder=1)
            
            key_mag = int(np.round(dbwind['wind_speed'].mean()/100., decimals=1)*100.) #round to closest 10
            plt.quiverkey(qp, 0.95, 1.025, key_mag, str(key_mag) +' Knots', fontproperties={'weight': 'bold'}, labelpos='W')
            axs[i].set_xlabel('Altitude layer: ' + str(Alayers[i]) + ' - ' + str(Alayers[i+1]) + ' ft.')
            axs[i].set_title(start + ' - ' + ctime)
            cb1 = plt.colorbar(cp, fraction=0.02, pad=0.08, cmap='Reds',ax=axs[i])
            cb1.set_label('Mean EDR Value ')          
            cb2 = plt.colorbar(sp, fraction=0.02, pad=0.04, ax=axs[i], shrink=1.25)
            cb2.set_label('')
        
        return db
 



def time_transform(tin, *, Back=False):
    '''
    simple routine to transform between observation time string and utc seconds
    By default goes from string to seconds, keyword back will do the opposite
    tin format = '2019-08-04 12:00:00+00'!!!
    '''
    import datetime
    
    if (Back == True):
        tout = datetime.datetime.utcfromtimestamp(tin).strftime('%Y-%m-%d %H:%M:%S+%z')
        tout = tout + '00' #adds utc timezone minutes column for strptime
    else:
        tin = tin + '00'  #adds utc timezone minutes column for strptime
        if (len(tin) > 24):
            tout = datetime.datetime.strptime(tin,'%Y-%m-%d %H:%M:%S.%f%z' )
        else:
            tout = datetime.datetime.strptime(tin,'%Y-%m-%d %H:%M:%S%z')
        
        tout = (tout - datetime.datetime(1970,1,1, tzinfo=datetime.timezone.utc))/datetime.timedelta(seconds=1)
        #tout = tout[0:-1]
        
    return tout
    
    
    
def create2d_data(db, key, latlongrid, *, Interp=False, resolution=1., Nanzero=False, Convolve=None):
    '''
    a function to interpolate mean data given keyword defined dataset 
    onto an evenly spaced latlongrid
    Inputs:
        db - database
        key - database keyword eg. 'temperature' of data you want to interpolate
              if you want wind vectors set to 'wind_vector' which will do both wind_speed and wind_direction
        latlongrid - latitude longitude grid you want to interpolate onto ; [lon1, lat1, lon2, lat2]
        resolution - tbd
        Nanzero - set to True to return zero points as nan
    Outputs:
        2d interpolated dataset..
    '''
    
    import numpy as np
    import metpy.calc as mp
    from scipy import interpolate
    
    #number of lat and lon bins
    nlat = int((latlongrid[3]-latlongrid[1])/resolution)
    nlon = int((latlongrid[2]-latlongrid[0])/resolution)

    #arrays of lat lon bin boundaries
    latspace = np.linspace(latlongrid[1], latlongrid[3], nlat+1)
    lonspace = np.linspace(latlongrid[0], latlongrid[2], nlon+1)
 
    #interpolating wind_direction requires breaking down into componenets and back, need extra dimension
    windvecs = 1 
    if (key == 'wind_vector'): windvecs=2
    
    #grid for binning mean variables into
    grid = np.zeros((nlon+1, nlat+1, windvecs))
   
    #looping and binning over lon and lat
    for i in range(nlon):
        for j in range(nlat):
            #selecting points in bin range
            points = db[ (db['longitude'] >= lonspace[i]) & (db['longitude'] <= lonspace[i+1]) \
                        & (db['latitude'] >= latspace[j]) & (db['latitude'] <= latspace[j+1]) ]
            
            if (len(points)>0):
                    
                if (key == 'wind_vector'): #wind vector handling
                    Ev, Nv = mp.wind_components(np.array(points['wind_speed']),\
                                                np.radians(np.array(points['wind_direction'])))
                    mEv = np.mean(Ev)
                    mNv = np.mean(Nv)
                    grid[i,j,0] = mEv
                    grid[i,j,1] = mNv
                else: #1d data handling
                    
                    mV = points[key].mean()
                    grid[i,j,0] = mV
     
    nlonsp = lonspace #+ resolution/2.
    nlatsp = latspace #+ resolution/2.               
    
    if (Interp==False) & (Convolve==None):
        
        print( 'No Interpolation, No Convolution')
        if (Nanzero==True): grid[np.where(grid == 0)] = np.nan      
        if (key == 'wind_vector'):
            return nlatsp, nlonsp, grid
        else:
            return nlatsp, nlonsp, grid[:,:,0]
    
    if (Convolve!=None):
        grid[np.where(grid == 0)] = np.nan
        grid[:,:,0] = convolve_nans(grid[:,:,0], stddev=Convolve)        
        
        print ('Convolution, No Interpolation')
        
        if (key == 'wind_vector'):
            return nlatsp, nlonsp, grid
        else:
            return nlatsp, nlonsp, grid[:,:,0]
    
    if (Interp==True):  
        #interpolating
        print('Interpolation')
        if (key == 'wind_vector'):
            #the ordering of lat and lon is wacky because interp2d is weird, should work out... check!
            f = interpolate.interp2d(nlatsp, nlonsp, grid[:,:,0], kind='cubic')
            g = interpolate.interp2d(nlatsp, nlonsp, grid[:,:,1], kind='cubic')
        else:
            f = interpolate.interp2d(nlatsp, nlonsp, grid[:,:,0], kind='linear')
            
        #define new grids to interpolate onto, at the moment just double the resolution idk..
        flonsp =  np.linspace(latlongrid[0], latlongrid[2], 2*nlat+1)+ resolution/4.
        flatsp =  np.linspace(latlongrid[1], latlongrid[3], 2*nlon+1) + resolution/4.
             
        igrid = np.zeros((len(flonsp), len(flatsp), windvecs))  

        if (key == 'wind_vector'):
            igrid[:,:,0] = f(flonsp, flatsp)
            igrid[:,:,1] = g(flonsp, flatsp)
            if (Nanzero==True): igrid[np.where(igrid == 0)] = np.nan 
            return flatsp, flonsp, igrid
            
        else:
            igrid[:,:,0] = f(flonsp, flatsp)
            if (Nanzero==True): igrid[np.where(igrid == 0)] = np.nan 
            return flatsp, flonsp, igrid[:,:,0]

    
    
def grid_db(db, key, latlongrid, *, resolution=2.):
    '''
    function to grid up oversampled point data into evenly spaced points for plotting
    Inputs:
        db - database
        key - database keyword eg. 'temperature' of data you want to interpolate
              if you want wind vectors set to 'wind_vector' which will do both wind_speed and wind_direction
        latlongrid - latitude longitude grid you want to interpolate onto ; [lon1, lat1, lon2, lat2]
        resolution - tbd
    Outputs:
        2d interpolated dataset..
    '''
    
    import numpy as np
    import metpy.calc as mp
    import pandas as pd
    
    #number of lat and lon bins
    nlat = int((latlongrid[3]-latlongrid[1])/resolution)
    nlon = int((latlongrid[2]-latlongrid[0])/resolution)
    
    #arrays of lat lon bin boundaries
    latspace = np.linspace(latlongrid[1], latlongrid[3], nlat+1)
    lonspace = np.linspace(latlongrid[0], latlongrid[2], nlon+1)     
    #initate output arrays     

    lat = np.array([])
    lon = np.array([])
    
    if (key == 'wind_vector'):
        Eva = np.array([])
        Nva = np.array([])
        Ws = np.array([])
    else:
        var = np.array([])
    count = np.array([])
        
    #looping and binning over lon and lat
    for i in range(nlon):
        for j in range(nlat):
            #selecting points in bin range
            points = db[ (db['longitude'] >= lonspace[i]) & (db['longitude'] <= lonspace[i+1]) \
                        & (db['latitude'] >= latspace[j]) & (db['latitude'] <= latspace[j+1]) ]
            
            lon = np.append(lon, lonspace[i]+resolution/2.)  #centre of bin
            lat = np.append(lat, latspace[j]+resolution/2.)
            count= np.append(count, len(points))
            
            if (key == 'wind_vector'):
                #wind direction needs special treatment as i cant take the mean angle really...
                if (len(points) > 0):
                    Ev, Nv = mp.wind_components(np.array(points['wind_speed']),\
                                                    np.radians(np.array(points['wind_direction'])))
                else:
                    Ev = 0
                    Nv = 0
                
                Eva = np.append(Eva, np.mean(Ev))
                Nva = np.append(Nva, np.mean(Nv))
                Ws = np.append(Ws, points['wind_speed'].mean())
                    
            else:
                if (len(points) > 0):
                    varm = points[key].mean()
                else:
                    varm=0
                var = np.append(var,varm)            
    
    if (key == 'wind_vector'):
        Eva[np.where(Eva == 0)] = np.nan
        Nva[np.where(Eva == 0)] = np.nan    
        Ws[np.where(Ws == 0)] = np.nan
        count[np.where(count==0)] = np.nan
        
        outdb = pd.DataFrame(index=[i for i in range(len(lon))])
        outdb['latitude'] = pd.Series(lat).values
        outdb['longitude'] = pd.Series(lon).values
        outdb['EW_velocity'] = pd.Series(Eva).values
        outdb['NS_velocity'] = pd.Series(Nva).values
        outdb['wind_speed'] = pd.Series(Ws).values
        outdb['count'] = pd.Series(count).values
       
    else:
        outdb = pd.DataFrame(index=[i for i in range(len(lon))])
        outdb['latitude'] = pd.Series(lat).values
        outdb['longitude'] = pd.Series(lon).values
        outdb[key] = pd.Series(var).values
        outdb['count'] = pd.Series(count).values

    return outdb   
    

    
def plot_latlon_scatter(DataFrame, *, latlongrid=[-180, -75, 180, 75], ckey='edr_mean_value'):
     
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    
    lats = DataFrame['latitude']
    lons = DataFrame['longitude']

    fig, axs = plt.subplots(1,1,figsize=(12,12))
    
    
    bm = Basemap(resolution='l', epsg=3857, ax=axs, \
                 llcrnrlon=latlongrid[0] ,\
                 llcrnrlat=latlongrid[1] ,\
                 urcrnrlon=latlongrid[2] ,\
                 urcrnrlat=latlongrid[3])
    
    
    bm.drawcoastlines()
    bm.fillcontinents(alpha=0.3, color='green')
    bm.drawcountries()
    

    
    scatterplot = bm.scatter(np.array(lons), np.array(lats), latlon=True, c=list(DataFrame[ckey]),
                             marker='+', cmap='viridis', ax=axs)
   
    cb=plt.colorbar(scatterplot, fraction=0.02, pad=0.04, cmap='viridis')
    cb.set_label(ckey)
    
    return DataFrame
    
    
def gradients(start, end, *, key='temperature', latlongrid=[-140, 15, -50, 65], Alayers=[0,50000], \
              resolution=2., grid=False, Interp=False, Convolve=None, OutA=False):
    
    '''
    function to plot variable gradients and turbulence reports together
    Inputs:
        start - start time string - '2019-08-01 12:00:00+00'
        end - end time string - "" ""
        key - set to key of what you want the gradient plot to be of...
    output:
        plot and some stats
        
        
    '''
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt

    db = get_db(start, end, latlongrid=latlongrid, Alayer=[0,50000])


    if (OutA==False):
        nlayers = len(Alayers)-1
        nrows = 2
        fig, axs = plt.subplots(nlayers,nrows,figsize=(nlayers*24,nrows*36))
        if (nlayers + nrows > 2): axs=axs.flatten()
        if (nlayers + nrows ==2): axs = [axs]
    
        i=0
        for j in range(nlayers):
            print('###', i, i+1)
            
            dbf = db
            
            dbf = dbf[(dbf['altitude'] >= Alayers[j]) & (dbf['altitude'] <= Alayers[j+1])]
            
            bm1 = Basemap(resolution='l', epsg=3857, ax=axs[i], \
                         llcrnrlon=latlongrid[0] ,\
                         llcrnrlat=latlongrid[1] ,\
                         urcrnrlon=latlongrid[2] ,\
                         urcrnrlat=latlongrid[3] )        
        
            bm1.drawcoastlines()
            bm1.fillcontinents(alpha=0.25, color='green')
            bm1.drawcountries()  
      
            bm2 = Basemap(resolution='l', epsg=3857, ax=axs[i+1], \
                 llcrnrlon=latlongrid[0] ,\
                 llcrnrlat=latlongrid[1] ,\
                 urcrnrlon=latlongrid[2] ,\
                 urcrnrlat=latlongrid[3] )        
        
            bm2.drawcoastlines()
            bm2.fillcontinents(alpha=0.25, color='green')
            bm2.drawcountries()  
                    
            if (grid==True):
                griddb = grid_db(dbf, 'edr_mean_value', latlongrid, resolution=2.)
                lats = np.array(griddb['latitude'])
                lons = np.array(griddb['longitude'])
                clist = list(griddb['edr_mean_value'])
                slist = list(griddb['edr_mean_value']*500.)
            else:
                dbscatter = dbf[dbf['edr_mean_value'] > 0.06]
                lats = np.array(dbscatter['latitude'])
                lons = np.array(dbscatter['longitude'])
                clist = list(dbscatter['edr_mean_value'])
                slist = list((dbscatter['edr_mean_value']+0.01)*500.)
            
            #making contours
            
            if (key == 'wind_vector'):
                xc, yc, cont2d = create2d_data(dbf, 'wind_vector', latlongrid, resolution=resolution, Nanzero=True, \
                                               Interp=Interp, Convolve=Convolve)
                xx, yy = np.meshgrid(xc, yc)
                
                #calculationg gradient taking into account wind direction.... i think
                cgEw = np.gradient(cont2d[:,:,0], yc, xc, edge_order=2)
                cgEw[0][np.where(cgEw[0] == np.nan)] = 0
                cgEw[1][np.where(cgEw[1] == np.nan)] = 0
                cgEw = np.sqrt(cgEw[0]**2.0 + cgEw[1]**2.0)
                
                cgNs = np.gradient(cont2d[:,:,1], yc, xc, edge_order=2)
                cgNs[0][np.where(cgNs[0] == np.nan)] = 0
                cgNs[1][np.where(cgNs[1] == np.nan)] = 0            
                cgNs = np.sqrt(cgNs[0]**2.0 + cgNs[1]**2.0)
                
                cg = np.sqrt(cgEw**2.0 + cgNs**2.0)
                minmax = np.nanmax(np.abs(cg))          
                #cant contour plot a [:,:,2] shaped array, so i will simply plot wind magnitude alongside gradient
                #gradient should have wind direction accounted for
                #cont2d = np.sqrt(cont2d[:,:,0]**2.0 + cont2d[:,:,1]**2.0)   
    
                #or quiver plot?
                
                #PLOTTING -----------------------------------------------------
                              
                contour_cmap = 'hot'
                gradient_cmap = 'Reds'
                
                #basemap1 - plots wind quiver plot
                dbwind = grid_db(dbf, 'wind_vector', latlongrid, resolution=resolution)
                
                
                qp1 = bm1.quiver(np.array(dbwind['longitude']), np.array(dbwind['latitude']), \
                                 np.array(dbwind['EW_velocity']), np.array(dbwind['NS_velocity']), latlon=True, \
                                 width=0.003, pivot='tail', color='blue', zorder=0)
                
                key_mag = int(np.round(dbwind['wind_speed'].mean()/100., decimals=1)*100.)
                axs[i].set_title(' ')
                plt.quiverkey(qp1, 0.95, 1.025, key_mag, str(key_mag) + ' Knots', fontproperties={'weight': 'bold'}, labelpos='W')                
                
                
                sp1 = bm1.scatter(lons,lats, latlon=True, s=slist,\
                                  marker='o', color='black', zorder=1)        
                handles, labels = sp1.legend_elements(prop='sizes')
                axs[i].legend([handles[-1]], [str(np.round(np.max(clist), decimals=2))], loc="lower left", title='Mean EDR')
               
                cb1b = plt.colorbar(sp1, fraction=0.02, pad=0.08, ax=axs[i])
                cb1b.set_label('EDR mean value')
                
                
    
                #basemap2 - plots gradient contour of key variable
                cp2 = bm2.pcolormesh(yy, xx, cg, cmap=gradient_cmap, zorder=0, latlon=True, vmin=0, vmax=minmax)
                bm2.scatter(lons, lats, latlon=True, s=slist ,\
                                 marker='o', color='black', zorder=1)
                axs[i+1].legend([handles[-1]], [str(np.round(np.max(clist), decimals=2))], loc="lower left", title='Mean EDR')
                       
                cb2a = plt.colorbar(cp2, fraction=0.02, pad=0.08,ax=axs[i+1])
                cb2a.set_label(key + ' Gradient')
    
                axs[i].set_title(start + ' --  ' + end)
                axs[i].set_xlabel('Measured wind vectors: ' + str(Alayers[i]) + ' - ' + str(Alayers[i+1]) + ' ft.')
                axs[i+1].set_title(start + ' -- ' + end)
                axs[i+1].set_xlabel('Calculated ' + key + ' Gradient ')         
    
                
                i += 2
                                       
            else:
                
    
                contour_cmap = 'hot'
                gradient_cmap = 'Reds'
                
                xc, yc, cont2d = create2d_data(dbf, key, latlongrid, resolution=resolution, Nanzero=True, Interp=Interp, Convolve=Convolve)
                xx, yy = np.meshgrid(xc, yc)
                    
                #calculate gradient
                        
                cg = np.gradient(cont2d, yc, xc, edge_order=2)
                cg[0][np.where(cg[0] == np.nan)] = 0
                cg[1][np.where(cg[1] == np.nan)] = 0 
                cg = np.sqrt(cg[0]**2.0 + cg[1]**2.0)        
                
                minmax = np.nanmax(np.abs(cg))
    
            
                #PLOTTING -----------------------------------------------------
                #basemap1 - plots just key contour
                sp1 = bm1.scatter(lons,lats, latlon=True, s=slist ,\
                                  marker='o', color='black', zorder=1) 
                
                handles, labels = sp1.legend_elements(prop='sizes')
                axs[i].legend([handles[-1]], [str(np.round(np.max(clist), decimals=2))], loc="lower left", title='Mean EDR')
                cp1 = bm1.pcolormesh(yy, xx, cont2d, cmap=contour_cmap, zorder=0, latlon=True, alpha=0.6)                     
    
                cb1a = plt.colorbar(cp1, fraction=0.02, pad=0.08,ax=axs[i])
                cb1a.set_label(key)
                
                #basemap2 - plots gradient contour of key variable
                cp2 = bm2.pcolormesh(yy, xx, cg, cmap=gradient_cmap, zorder=0, latlon=True, vmin=0, vmax=minmax)
                bm2.scatter(lons, lats, latlon=True, s=slist ,\
                                 marker='o', color='black', zorder=1)
                axs[i+1].legend([handles[-1]], [str(np.round(np.max(clist), decimals=2))], loc="lower left", title='Mean EDR')
                  
    
                cb2a = plt.colorbar(cp2, fraction=0.02, pad=0.08, cmap='Reds',ax=axs[i+1])
                cb2a.set_label(key + ' Gradient')
    
                axs[i].set_title(start + ' --  ' + end)
                axs[i].set_xlabel('Measured Temperature: ' + str(Alayers[i]) + ' - ' + str(Alayers[i+1]) + ' ft.')
                axs[i+1].set_title(start + ' -- ' + end)
                axs[i+1].set_xlabel('Calculated ' +  key + ' Gradient')
                
                
                #do the maths...
              
                
    
                    
                i += 2
    
    #return 2d histograms for stats
    if (OutA == True):
        dbf = db[(db['altitude'] >= Alayers[0]) & (db['altitude'] <= Alayers[-1])]
        xc, yc, contedr = create2d_data(dbf, 'edr_peak_value', latlongrid, resolution=resolution, Nanzero=True, Interp=Interp, Convolve=Convolve)      
        
        xc, yc, contkey = create2d_data(dbf, key, latlongrid, resolution=resolution, Nanzero=True, Interp=Interp, Convolve=Convolve)                
        #calculate gradient
        if (key=='wind_vector'):
             #calculationg gradient taking into account wind direction.... i think
            cgEw = np.gradient(contkey[:,:,0], yc, xc, edge_order=2)
            cgEw[0][np.where(cgEw[0] == np.nan)] = 0
            cgEw[1][np.where(cgEw[1] == np.nan)] = 0
            cgEw = np.sqrt(cgEw[0]**2.0 + cgEw[1]**2.0)
            
            cgNs = np.gradient(contkey[:,:,1], yc, xc, edge_order=2)
            cgNs[0][np.where(cgNs[0] == np.nan)] = 0
            cgNs[1][np.where(cgNs[1] == np.nan)] = 0            
            cgNs = np.sqrt(cgNs[0]**2.0 + cgNs[1]**2.0)
            
            cg = np.sqrt(cgEw**2.0 + cgNs**2.0)
        else:

            cg = np.gradient(contkey, yc, xc, edge_order=2)
            cg[0][np.where(cg[0] == np.nan)] = 0
            cg[1][np.where(cg[1] == np.nan)] = 0 
            cg = np.sqrt(cg[0]**2.0 + cg[1]**2.0)        

        
        
        return cg, contedr
    else:    
        return dbf
    


def convolve_nans(im, *, stddev=1):
    '''
    Convolve the data and replace nans with kernal-weighted interpolation from their neighbours
    Increase stddev to smooth more drastically..
    '''

    import astropy.convolution as apc # Gaussian2DKernel
                                      #convolve
    
    kernel = apc.Gaussian2DKernel(x_stddev=stddev)
    imout = apc.convolve(im, kernel, boundary='extend') 
    

    return imout
    
    
def gen_pdf(x, Noise=False):
    '''
    Generate and return a quick PDF plot and array, respectively for a given variable x.
    Set Noise to true to add random gaussian noise to the distribution - smooths edr binning for example
    '''
    import matplotlib.pyplot as plt
    import scipy as sp
    import numpy as np
    fig, axs = plt.subplots(1,1,figsize=(12,8))
    
    if (Noise==True):
        x = x + np.random.normal(0,1,len(x))/1000.
    
    kde = sp.stats.gaussian_kde(x)
    axs.plot(np.linspace(0,np.nanmax(x), 1000), kde(np.linspace(0,np.nanmax(x), 1000)), linewidth=5.0)
    axs.set_yscale('log')
    axs.set_ylim(bottom=0.001)
    axs.hist(x, 500, density=True)
    
    a = (sp.integrate.quad(kde, 0, np.nanmax(x)))
    axs.text(0.6*axs.get_xlim()[1], 0.6*axs.get_ylim()[1], 'Area: ' + str(a[0]), fontsize=10)
    
    
    #CDF from pandas series:
    #t=(edrp.value_counts().sort_index().cumsum())/np.nanmax(edrp.value_counts().sort_index().cumsum())
    return kde(np.linspace(0,np.nanmax(x), 100))
    
  
    
def grid_and_plot_data(db, latlongrid, *, resolution=2., key='edr_peak_value', ptype='pmesh'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import metpy.calc as mp
    
    '''
    ptype can be =pmesh, scatter, contour, hexbin
    
    
    
    '''
    
    
    ptypes = ['pmesh', 'scatter', 'contour', 'hexbin']
    if (ptype not in ptypes):
        print(ptype + ' is not a valid keyword! Use any of : [pmesh, scatter, contour, hexbin]')
        return db
    
    
    
    griddb = grid_db(db, key, latlongrid, resolution=resolution )
    x, y, db2d = create2d_data(db, key, latlongrid, resolution=resolution)
    xx, yy = np.meshgrid(x, y)
    
    fig, axs = plt.subplots(1,1, figsize=(12,12))
    bm = Basemap(resolution='l', epsg=3857, ax = axs,\
                         llcrnrlon=latlongrid[0] ,\
                         llcrnrlat=latlongrid[1] ,\
                         urcrnrlon=latlongrid[2] ,\
                         urcrnrlat=latlongrid[3] )
    bm.drawcoastlines()
    bm.fillcontinents(alpha=0.4, color='green')
    bm.drawcountries()
 

    if (ptype == 'pmesh'):
        cp = bm.pcolormesh(yy, xx, db2d, cmap='Reds', latlon=True, alpha=0.5, zorder=1)
        cb = bm.colorbar(cp, location='right', ax=axs)
        cb.set_label(key)
    
    if (ptype == 'scatter'):
        sp = bm.scatter(np.array(griddb['longitude']), np.array(griddb['latitude']), latlon=True, s=list(griddb[key]*250.), \
                        marker='o', c='k', zorder=2)
    
    if (ptype == 'contour'):
        cp = bm.contour(yy, xx, db2d, cmap='Reds', latlon=True, alpha=0.5, zorder=1)
        cpf = bm.contourf(yy, xx, db2d, cmap='Reds', latlon=True, alpha=0.5, zorder=1)
        cb = bm.colorbar(cpf, location='right', ax=axs)
        cb.set_label(key)
        
#    if (ptype == hexbin):

        
    
    return db
    
    

def common_pairs(dbt, *, DAL=False, dist=False):
    '''
    Routine to produce the most common airport pair in a dataframe
    Set DAL to true to exclude Delta
    '''
    import numpy as np
    import pandas as pd
    
    if (DAL==True): dbt = dbt[dbt['flight_airline'] != 'DAL'] #until i fix delta destination airports code...


    #reduce dataframe to unique flights and their departure/destination with no nans
    dbt = dbt.drop_duplicates(subset='metadata_tafi', keep='first')
    dbt = dbt.dropna(subset=['flight_departure_aerodrome', 'flight_destination_aerodrome'])
    
    #big fat list of all the aerodrome pairs for reduction
    pairs = np.array( (dbt['flight_departure_aerodrome'] + dbt['flight_destination_aerodrome']))
    #if it finds a reversed pair it flips it so we count both directions as a single route
    print(len(pairs))
    for i in range(len(pairs)): #minus 1 to skip nans 
        if (i%np.round(len(pairs)/10.) == 0): print(np.round((i/len(pairs))*100.), ' %') #percentage to keep track of progress...
        test = pairs[i] #first pair
        s = np.where(pairs == (test[4:8] + test[0:4]))[0] #check for all reverses 
        pairs[s] = test #changes reverse pairs to same orientation as original 
    
    
    
    mc = pd.DataFrame(columns=['pairs', 'count', 'distance'])
    pS = pd.Series(pairs)
    vc = pS.value_counts()
    
    mc['pairs'] = pd.Series(vc.index.values)
    mc['count'] = pd.Series(pS.value_counts().values)
     
    if (dist==True):
           from haversine import haversine
           ap = pd.read_csv('~/Documents/CSV/airport-locations.csv')
           pu = mc['pairs'].unique()
           distance = np.zeros(len(pu))
           for j in range(len(pu)):
                  lat1, lat2 = ap[ap['airport_code'] == pu[j][0:4]]['lat'], ap[ap['airport_code'] == pu[j][4:8]]['lat']
                  lon1, lon2 = ap[ap['airport_code'] == pu[j][0:4]]['lon'], ap[ap['airport_code'] == pu[j][4:8]]['lon']
                  
                  #some AP not in the database
                  if (len(lat1) ==1) & (len(lat2)==1) & (len(lon1)==1) & (len(lon2)==1):
                         distance[j] = haversine((lat1.iloc[0],lon1.iloc[0]), (lat2.iloc[0],lon2.iloc[0])) #bullshit way around but thats how the function is coded zzzzz... (lon lat for lyfe)
                  
           
           mc['distance'] = distance
    '''           
    print('Most common aerodrome pair: ' + get_AP_name(mcn[0:4]) + ':' + get_AP_name(mcn[4:8]) \
          + ' - ' + str(pd.Series(pairs).value_counts()[0]) + ' flights.')
    '''
    
    return mc
     
        
        
def get_Pdb(dbt, edr, *, altitude=[0,50000], resolution=2., Interpolate=False, ptype='pmesh'):

    #llcrnrlon=-180.,llcrnrlat=-65,urcrnrlon=180.,urcrnrlat=75
    '''
    Function to produce a interpolated 2darray in lat and lon axis
    of % chance of encountering a turbulent event within given paraneters.
    Inputs:
        ############################ IMPORTANT #######################################
        dbt - a dataframe that has had the flights interpolated onto 60. intervals!!!!!
        edr - EDR threshold that you want to flag as a turbulent event
                    This is a complex threshold, leave at 0.18 for now... 
        ##############################################################################
        altitude - altitude regions you want to filter between [floor, ceiling]
        binsize - resolution of grid - passed to other functions
    '''
    import pandas as pd
    import numpy as np
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    
    #do the interpolation here otherwise   
    if (Interpolate == True):
        dbt = interpolate_db(dbt)
        
    #filter altitude
    dbt=dbt[(dbt['altitude']>altitude[0]) & (dbt['altitude']<altitude[1])]
    
    #define a database of turbulent events only
    db_event = dbt[dbt['edr_mean_value'] > edr]
    
    #get lat and lon grid size
    
    latlongrid = np.round([dbt['longitude'].min(), dbt['latitude'].min(), dbt['longitude'].max(), dbt['latitude'].max()])
    
    dball_grid = grid_db(dbt, 'edr_mean_value', latlongrid, resolution=resolution)
    dbeve_grid = grid_db(db_event, 'edr_mean_value', latlongrid, resolution=resolution)
    
    #dbeve_grid['count'].iloc[np.where(dbeve_grid['count'] == dball_grid['count'])] = 0. #removing small number statistic cases
    dbeve_grid = dbeve_grid[dball_grid['count'] >= (dball_grid['count'].mean() - dball_grid['count'].std()) ] # rremoving small number stats
    
    percent = (dbeve_grid['count']/dball_grid['count'])*100.
    
    Pdb = pd.DataFrame(columns=['longitude','latitude','percent'])
    Pdb['longitude'] = dball_grid['longitude']
    Pdb['latitude'] = dball_grid['latitude']
    Pdb['percent'] = pd.Series(percent)
    
    
    if (ptype=='kde'):
           fig, axs = plt.subplots(1,1, figsize=(10,10))
           bm = Basemap(resolution='l', epsg=3857, ax = axs,\
                        llcrnrlon=latlongrid[0] ,\
                        llcrnrlat=latlongrid[1] ,\
                        urcrnrlon=latlongrid[2] ,\
                        urcrnrlat=latlongrid[3] )
           bm.drawcoastlines()
           bm.fillcontinents(alpha=0.4, color='green')
           bm.drawcountries()
           
           nbins=100
           xall, yall = bm(np.array(dbt['longitude']), np.array(dbt['latitude']))
           x0, y0 = bm(latlongrid[0], latlongrid[1])
           x1, y1 = bm(latlongrid[2], latlongrid[3])
           xbins = np.linspace(x0,x1, nbins)
           ybins = np.linspace(y0,y1, nbins)           
           hall, xedges, yedges = np.histogram2d(xall,yall,bins=[xbins, ybins])
           #hall[hall < (np.mean(hall))] = 0
           xeve, yeve = bm(np.array(db_event['longitude']), np.array(db_event['latitude']))
           heve, xedges, yedges = np.histogram2d(xeve,yeve,bins=[xbins, ybins])          


           hper = (heve/hall)*100.
           hper[np.isnan(hper)] = 0. #remove nans from dividing by zero in some bins
                   
           import scipy.ndimage.filters
                      

           z = scipy.ndimage.filters.gaussian_filter((hper.T), resolution)
           z.shape
           im = bm.imshow(z, cmap='Reds', ax=axs, vmin=0, vmax=1.5*np.max(z))
           cb = bm.colorbar(im, location='right', ax=axs)
           cb.set_label('Percentage')
    else:
           test = grid_and_plot_data(Pdb, latlongrid, key='percent', resolution=resolution, ptype=ptype)
    
    return Pdb
    
    
    
    
def interpolate_flight(flightdf,*, interval=60.):
    '''Function to interpolate routine reports onto the same timescale as event reports, so that the percentage stats
        don't get fucked up by samping differences
        Input:
            flightdf - dataframe of the flight you want to interpolate
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    
    variables = ['utc_timestamp', 'altitude', 'longitude', 'latitude', 'temperature', 'wind_direction', \
             'wind_speed', 'edr_peak_value', 'edr_mean_value', 'report_flag']

    if len(flightdf) >=2:
            
        #THIS IS NOT A PERFECT SOLUTION - LEADS TO SOME ROUNDING ERRORS DEPENDING ON WHERE THE INTERPOLATION
        #FALLS IN THE TURBULENT INTERVALS. BUT ITS VERY SIMPLE AND RELATIVELY FAST AND THE ALTERNATIVES ARE COMPLICATED...
        #SO I AM LEAVING IT LIKE THIS FOR NOW.
        xn = np.arange(flightdf['utc_timestamp'].iloc[0], flightdf['utc_timestamp'].iloc[-1], interval)
        

        
        db = pd.DataFrame(columns=variables)
        db['utc_timestamp'] = xn
        for i in range(len(variables)):
            f = sp.interpolate.interp1d(flightdf['utc_timestamp'], flightdf[variables[i]])
            interp = f(xn)
            if (variables[i] == 'report_flag'): interp[interp<0.9]=0.
            db[variables[i]] = pd.Series(interp)
        
        
        #fig, axs = plt.subplots(2,1, figsize=(12,10))
        #axs[0].plot(flightdf['utc_timestamp'], flightdf['edr_mean_value'], marker='o')
        #axs[1].plot(db['utc_timestamp'], db['edr_mean_value'], marker='o')
        
        
        
        flightdf = db
    else:
        flightdf = flightdf[variables] #drop variables if above if statement isnt exacuted
    return flightdf
    
    
    
def interpolate_db(dbt, *, interval=60.):
    '''
    Wrapper around interpolate_flight to do it to an entire database
    '''
    import numpy as np
    import pandas as pd
    
    print('###WARNING### : Interpolation removes variables like tafi, destination etc.!!!! ')
    
    tafis = dbt['metadata_tafi'].dropna().unique()
    variables = ['utc_timestamp', 'altitude', 'longitude', 'latitude', 'temperature', 'wind_direction', \
         'wind_speed', 'edr_peak_value', 'edr_mean_value', 'report_flag']
    
    dbtemp = pd.DataFrame(columns=variables)
    
    print('Interpolating....')
    for i in range(len(tafis)):
        f = interpolate_flight(dbt[dbt['metadata_tafi'] == tafis[i]], interval=interval)
        dbtemp = pd.concat([dbtemp, f])
        if (i%np.round(len(tafis)/10.) == 0): print(np.round((i/len(tafis))*100.), ' %')
    
    dbt = dbtemp      
    
    return dbt
    
        
def city_pair(db, depA, desA, *, Plot=True, nbins=100, latlongrid=None, resolution=2.):
    '''
    Function to return (and plot) the mean flight path between two airport, depA and desA
    '''
    
    import pandas as pd
    import numpy as np
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    
    
    dbi=db
    
    db = find_flights(db, depA, desA, Both=True)
    
    tafis = db['metadata_tafi'].dropna().unique()
    ap = pd.read_csv('~/Documents/CSV/airport-locations.csv')
    
    nbins=nbins
    meanlons = np.zeros(nbins)
    meanlats = np.zeros(nbins)
        
    ap1lat = (ap[ap['airport_code']==depA])['lat']
    ap1lon = (ap[ap['airport_code']==depA])['lon']
    
    ap2lat = (ap[ap['airport_code']==desA])['lat']
    ap2lon = (ap[ap['airport_code']==desA])['lon']
    
    if (Plot==True):
           
           
        try:
               latlongrid=latlongrid
        except NameError:
               margin = 5.
               latlongrid = np.round([db['longitude'].min()-margin, db['latitude'].min()-margin, \
                                      db['longitude'].max()+margin, db['latitude'].max()+margin])
        
        fig, axs = plt.subplots(2,1,figsize=(18,12))
        
        #ONE######################################################################
        bm1 = Basemap(resolution='l', epsg=3857, ax=axs[0], \
                     llcrnrlon=latlongrid[0] ,\
                     llcrnrlat=latlongrid[1] ,\
                     urcrnrlon=latlongrid[2] ,\
                     urcrnrlat=latlongrid[3] )
        bm1.drawcoastlines()
        bm1.fillcontinents(alpha=0.4, color='green')
        bm1.drawcountries()
        
        sp1 = bm1.scatter(np.array(db['longitude']), np.array(db['latitude']), latlon=True, c=np.array(db['utc_timestamp']), \
                         cmap='viridis', marker='o', zorder=2, edgecolor='k', s=100.)
        
        x, y = bm1(ap1lon.iloc[0], ap1lat.iloc[0])
        axs[0].text(x,y, depA, bbox=dict(facecolor='white', alpha=1.0), zorder=5, ha='center', va='center')
        x, y = bm1(ap2lon.iloc[0], ap2lat.iloc[0])
        axs[0].text(x,y, desA, bbox=dict(facecolor='white', alpha=1.0), zorder=5, ha='center', va='center')
        #axs.text(ap2lon.iloc[0], ap2lon.iloc[0]+2.5, desA, latlon=True)
        
        #kde plot
        db_event= dbi[dbi['edr_mean_value'] > 0.06]
        xall,yall = bm1(np.array(dbi['longitude']), np.array(dbi['latitude']))
        x0,y0 = bm1(latlongrid[0], latlongrid[1])
        x1,y1 = bm1(latlongrid[2], latlongrid[3])
        xbins = np.linspace(x0,x1,nbins)
        ybins = np.linspace(y0,y1,nbins)
        hall, xedges, yedges = np.histogram2d(xall, yall, bins=[xbins,ybins])
       
        xeve, yeve = bm1(np.array(db_event['longitude']), np.array(db_event['latitude']))
        heve, xedges, yedges = np.histogram2d(xeve, yeve, bins=[xbins,ybins])
       
        hper = (heve/hall)*100.
        hper[np.isnan(hper)] =0.
       
        import scipy.ndimage.filters as snf
        z = snf.gaussian_filter((hper.T), resolution)
       
        im = bm1.imshow(z, cmap='Reds', vmin=0, vmax=1.5*np.max(z), zorder=0)
        bm1.colorbar(im, label='Percentage')
        #cb.set_label('Percentage')  
        
        #scattercolorbar
        cbs = bm1.colorbar(sp1, location='bottom', ax=axs[0])
        cticks = cbs.get_ticks()
        clabels = [time_transform(cticks[i], Back=True)[11:19] for i in range(len(cticks))]
        cbs.set_label('Time (UT)')
        cbs.set_ticks(cticks)
        cbs.set_ticklabels(clabels)
        
        #TWO######################################################################
        
        bm2 = Basemap(resolution='l', epsg=3857, ax=axs[1], \
                     llcrnrlon=latlongrid[0] ,\
                     llcrnrlat=latlongrid[1] ,\
                     urcrnrlon=latlongrid[2] ,\
                     urcrnrlat=latlongrid[3] )
        bm2.drawcoastlines()
        bm2.fillcontinents(alpha=0.4, color='green')
        bm2.drawcountries()
        
        sp2 = bm2.scatter(np.array(db['longitude']), np.array(db['latitude']), latlon=True, c=np.array(db['utc_timestamp']), \
                          cmap='viridis', marker='o', zorder=2, edgecolor='k', s=100.)       
        
               
        x, y, db2d = create2d_data(dbi, 'edr_mean_value', latlongrid, resolution=resolution) 
        xx, yy = np.meshgrid(x, y)
        
        cp = bm2.pcolormesh(yy, xx, db2d, cmap='Reds', latlon=True, alpha=0.5, zorder=0)
        bm2.colorbar(cp, label='Mean EDR Value')
        cbs2= bm2.colorbar(sp2, location='bottom', ax=axs[1])
        cticks = cbs.get_ticks()
        clabels = [time_transform(cticks[i], Back=True)[11:19] for i in range(len(cticks))]
        cbs2.set_label('Time (UT)')
        cbs2.set_ticks(cticks)
        cbs2.set_ticklabels(clabels)
        
        x, y = bm1(ap1lon.iloc[0], ap1lat.iloc[0])
        axs[1].text(x,y, depA, bbox=dict(facecolor='white', alpha=1.0), zorder=5, ha='center', va='center')
        x, y = bm1(ap2lon.iloc[0], ap2lat.iloc[0])
        axs[1].text(x,y, desA, bbox=dict(facecolor='white', alpha=1.0), zorder=5, ha='center', va='center')

        
    for tn in range(len(tafis)):
        flight = db[db['metadata_tafi'] == tafis[tn]]
                
        lons = np.array(flight['longitude'])
        lats = np.array(flight['latitude'])
        
        #add final points of airports
        lons = np.insert(lons, 0, ap1lon.iloc[0])
        lons = np.append(lons, ap2lon.iloc[0])
        
        lats = np.insert(lats, 0, ap1lat.iloc[0])
        lats = np.append(lats, ap2lat.iloc[0])
      
        lats = np.interp(np.linspace(lats[0], lats[-1], nbins), lats, lats, period=180) 
        lons = np.interp(np.linspace(lons[0], lons[-1], nbins), lons, lons, period=360)
           
        meanlons = meanlons + lons
        meanlats = meanlats + lats            
        

        if (flight['flight_departure_aerodrome'].iloc[0] == desA):
               linest=':'
        else:
               linest = '-'
        
        bm1.plot(np.array(flight['longitude']), np.array(flight['latitude']), latlon=True, c='k', alpha=0.2, zorder=1, linestyle=linest)
        bm2.plot(np.array(flight['longitude']), np.array(flight['latitude']), latlon=True, c='k', alpha=0.2, zorder=1, linestyle=linest)
        
        
    meanlons /=len(tafis)
    meanlats /=len(tafis)
    
    #if (Plot==True):
        #bm.scatter(meanlons, meanlats, latlon=True, marker='o', c='blue', alpha=0.6, zorder=1)
    
    
    return db, meanlons, meanlats
    

def airport_grid(A1, A2, *, factor=0.25):
    '''
    function to return a convenient latlongrid around airports of choice
    '''
    import pandas as pd
    import numpy as np
    ap = pd.read_csv('~/Documents/CSV/airport-locations.csv')
    
    ap1lat = (ap[ap['airport_code']==A1])['lat'].iloc[0]
    ap1lon = (ap[ap['airport_code']==A1])['lon'].iloc[0]
    
    ap2lat = (ap[ap['airport_code']==A2])['lat'].iloc[0]
    ap2lon = (ap[ap['airport_code']==A2])['lon'].iloc[0]
    
    lats = [ap1lat, ap2lat]
    lons = [ap1lon ,ap2lon]
    
    dlon = np.abs(ap1lon - ap2lon) * factor*0.5
    dlat = np.abs(ap1lat - ap2lat) * factor
    
    return np.round([np.min(lons)-dlon, np.min(lats)-dlat, np.max(lons)+dlon, np.max(lats)+dlat])
    
    
    
def perc_kde(db,bm,edr,*,nbins=100):
       '''
       Pass the db of interest and the Basemap instance and the edr event threshold
       '''

       import pandas as pd
       import numpy as np
       from mpl_toolkits.basemap import Basemap
       import matplotlib.pyplot as plt

       latlongrid = np.round([db['longitude'].min(), db['latitude'].min(), db['longitude'].max(), db['latitude'].max()])

       
       db_event= db[db['edr_mean_value'] > edr]
       xall,yall = bm(np.array(db['longitude']), np.array(db['latitude']))
       x0,y0 = bm(latlongrid[0], latlongrid[1])
       x1,y1 = bm(latlongrid[2], latlongrid[3])
       xbins = np.linspace(x0,x1,nbins)
       ybins = np.linspace(y0,y1,nbins)
       hall, xedges, yedges = np.histogram2d(xall, yall, bins=[xbins,ybins])
       
       xeve, yeve = bm(np.array(db_event['longitude']), np.array(db_event['latitude']))
       heve, xedges, yedges = np.histogram2d(xeve, yeve, bins=[xbins,ybins])
       
       hper = (heve/hall)*100.
       hper[np.isnan(hper)] =0.
       
       import scipy.ndimage.filters as snf
       z = snf.gaussian_filter((hper.T), 3)
       
       im = bm.imshow(z, cmap='Reds', vmin=0, vmax=1.5*np.max(z), zorder=0)
       cb = bm.colorbar(im, location='right')
       cb.set_label('Percentage')
       
       return
       

def investigate_airport(airport_code, start, end, *, Ashells=[0,  1000,  2000,  3000,  4000,  5000,  6000,  7000,  8000,\
        9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,\
       18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000,\
       27000], Dshells=[  0,  45,  90, 135, 180, 225, 270, 315, 360], edr=0.06, lshell=2000.):
       '''
       Function to plot turbulence occurent in and out of an airport
       airport_code- eg 'KJFK'
       start/end -  = eg 2019-08-01 12:00:00+00'
       
       edr - cutoff EDR for 'turbulence'
       lshell - cutoff lowest shell
       '''
       import numpy as np
       import pandas as pd
       import metpy.calc as mp
       from mpl_toolkits.basemap import Basemap
       import matplotlib.pyplot as plt       
       
       #setting up the database of relevant points
       ap = pd.read_csv('~/Documents/CSV/airport-locations.csv')
       c = [ap[ap['airport_code'] == airport_code]['lon'].iloc[0], ap[ap['airport_code'] == airport_code]['lat'].iloc[0]]
       grid = [np.round(c[0] -2.,decimals=1), np.round(c[1] -2.,decimals=1), np.round(c[0]+2.,decimals=1), np.round(c[1]+2.,decimals=1)]
       dbi = get_db(start, end, latlongrid=grid, Alayer=[0,40000])
       dbi = dbi[(dbi['flight_departure_aerodrome'] == airport_code) | (dbi['flight_destination_aerodrome'] == airport_code)]
       db = interpolate_db(dbi)
       R=1.0
       db = db[ (((db['longitude'] - c[0])**2 + (db['latitude'] - c[1])**2) <  R)]
       
       
       #plotting 
       #polar plot direction dependancy
       #using wind_direction because its a easy way to calculate angle from x, y - *minus 1 because wind direction convention is reversed
       headings = pd.Series(np.array(mp.wind_direction(np.array(db['longitude'] - c[0])*-1., \
                                                       np.array(db['latitude'] - c[1])*-1.)), \
                                                         name='headings', index=db.index)           
       DAarray = np.zeros((len(Dshells)-1, len(Ashells)-1))
       C = np.zeros_like(DAarray)

       fig = plt.figure(figsize=(10,10))
       ax = plt.subplot(111, projection='polar')
       ax.set_theta_direction(-1)
       ax.set_theta_zero_location('N')
       
       #create array       
       for j in range(len(Dshells)-1):
              for k in range(len(Ashells)-1):
                     dbarc = db[(headings > Dshells[j]) & (headings < Dshells[j+1]) &  \
                                (db['altitude'] > Ashells[k]) & (db['altitude'] < Ashells[k+1])]
                     
                     #dbslice = db[(headings > Dshells[j]) & (headings < Dshells[j+1])]
                     
                     if (len(dbarc) > len(db)*0.001) and (Ashells[k] >= lshell):
                            P = (len(dbarc[dbarc['edr_peak_value'] > edr])/len(dbarc))*100.
                     else:
                            P = np.nan
                     DAarray[j,k] = P
                     C[j,k] = len(dbarc)
     
       #plot array 
       ca = DAarray/np.nanmax(DAarray)
       for w in range(len(Dshells)-1):
              for x in range(len(Ashells)-1):
                     if np.isnan(ca[w,x]) == True:
                            ax.bar(np.radians(Dshells[w]), (Ashells[x+1] - Ashells[x]), bottom=x*1000., \
                                  align='edge', color='white')
                     else:                        
                            ax.bar(np.radians(Dshells[w]), (Ashells[x+1] - Ashells[x]), bottom=x*1000., \
                                   align='edge', color=plt.cm.viridis(ca[w,x]), edgecolor='k')
       


       ax.plot(np.linspace(0,2*np.pi, 100), np.zeros(100)+10000., color='white', linestyle='--', zorder=2)
       ax.plot(np.linspace(0,2*np.pi, 100), np.zeros(100)+20000., color='white', linestyle='--', zorder=2)
       #ticks = [str(int(ax.get_yticks()[i])) for i in range(len(ax.get_yticks()))]
       ax.set_rlabel_position(270.)
       ax.set_yticklabels(' ')
       ax.set_yticks([10,20])
       ax.set_title(get_AP_name(airport_code))
       
       for i in range(len(ax.get_yticks())):
              ax.text(np.radians(ax.get_rlabel_position()), ax.get_yticks()[i]*1000, 'FL'+str(int(ax.get_yticks()[i])), ha='center', va='center', bbox=dict(facecolor='white', alpha=1.), zorder=3 )
       
       ax.text(0,0,airport_code, ha='center', va='center', color='white',  bbox=dict(facecolor=plt.cm.viridis(0), alpha=1.))
       from matplotlib.cm import ScalarMappable
       cb = plt.colorbar(ScalarMappable(cmap='viridis', norm=plt.Normalize(0, np.nanmax(DAarray))), shrink=0.6)
       cb.set_label('Percentage, %')
       

       #lineplot
       #lineplot total %
       Ashells_lp = np.linspace(0,40000, 41)  #lineplot Ashells
       Pshells = np.zeros_like(Ashells_lp)
       
       DAlines = np.zeros((len(Dshells)-1, len(Ashells_lp)-1))
       C = np.zeros_like(DAlines)
       
       #crate another array similar to above but with better altitude resolution for the line plot
       for j in range(len(Dshells)-1):
              for k in range(len(Ashells_lp)-1):
                     dbarc = db[(headings > Dshells[j]) & (headings < Dshells[j+1]) &  \
                                (db['altitude'] > Ashells_lp[k]) & (db['altitude'] < Ashells_lp[k+1])]
                     
                     #dbslice = db[(headings > Dshells[j]) & (headings < Dshells[j+1])]
                     
                     if (len(dbarc) > len(db)*0.01) and (Ashells[k] >= lshell):
                            P = (len(dbarc[dbarc['edr_peak_value'] > edr])/len(dbarc))*100.
                     else:
                            P = np.nan
                     DAlines[j,k] = P
                     C[j,k] = len(dbarc)
       
       
       #plotting lines
       fig1 = plt.figure(figsize=(10,5))
       ax1 = plt.subplot(111)
       ax1.grid(which='major')
       ax1.set_xlabel('Flight level')
       ax1.set_ylabel('Turbulence encounter percentage')
       
       meanline = np.nanmean(DAlines, axis=0)   
             
       for l in range(len(Dshells)-1):
              ax1.plot(Ashells_lp[0:-1]/1000., DAlines[l,:], marker='o', color=plt.cm.viridis(l/len(Dshells)) )
       ax1.plot(Ashells_lp[0:-1]/1000., meanline, color='k', lw=5.0, ls='--')
       
       
       
       
       '''
       #ORIGINAL IDEA - INCLUDED IN CASE I WANNA GO BACK AND NOT WRITE IT ALL AGAIN...
       for j in range(len(Dshells)-1):
              for k in range(len(Ashells)-1):
                     dbarc = db[(headings > Dshells[j]) & (headings < Dshells[j+1]) & (db['altitude'] > Ashells[k]) & (db['altitude'] < Ashells[k+1])]
                     dbslice = db[(headings > Dshells[j]) & (headings < Dshells[j+1])]
                     if len(dbarc) > 1:
                            DAtest[j,k] = (len(dbarc[dbarc['edr_peak_value'] > edr])/len(dbslice))*100.
                     else:
                            DAtest[j,k] = 0.
              
              current_wedge = DAtest[j,:]
              bottom = 0.0
              colors = plt.cm.viridis([i/(len(current_wedge)) for i in range(len(current_wedge))])
              for w in range(len(current_wedge)):
                     ax.bar(np.radians(Dshells[j]), current_wedge[w], bottom=bottom, align='edge', color=colors[w], edgecolor='k')
                     bottom += current_wedge[w]
       '''
              

       
       return DAarray, C
   
       
       
       
       
    
    
    