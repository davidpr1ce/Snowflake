from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


db = past_conditions('2019-08-01 12:00:00+00', 6.0, Grid=False, resolution=1., Alayers=[30000, 40000], latlongrid=[-36, 51, -24, 54.5])
dbt = db[ (db['latitude'] >=51) & (db['latitude'] <= 54.5) & (db['longitude'] >= -36) & (db['longitude'] <= -24)]
#dbt = dbt[(dbt['utc_timestamp']> time_transform('2019-08-01 7:12:40+00')) & (dbt['utc_timestamp'] < time_transform('2019-08-01 10:18:00+00'))]
#dbne = dbt[(dbt['edr_peak_value'] < 0.1)]
#dbe = dbt[(dbt['edr_peak_value'] > 0.1)]
dbe = dbt[(dbt['edr_peak_value'] < 0.025)]



tafis = dbe['metadata_tafi'].unique()
fig1 = plt.figure(figsize=(16,9))
#ax = fig.add_subplot(111, projection='3d')
ax1 = fig1.add_subplot(121)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Altitude')

ax2 = fig1.add_subplot(122)
ax2.set_xlabel('Latitude')
ax2.set_ylabel('Altitude')

fig2 = plt.figure(figsize=(12,14))
ax3d = fig2.add_subplot(111, projection='3d')
ax3d.set_xlabel('Longitude')
ax3d.set_ylabel('Latitude')
ax3d.set_zlabel('Altitude')

#ax.set_zlabel('Altitude')
lon = np.array(dbe['longitude'])
lat = np.array(dbe['latitude'])
alt = np.array(dbe['altitude'])
edr = np.array(dbe['edr_peak_value'])    

#sp = ax.scatter(lon, lat, alt, color='r')
sp1 = ax1.scatter(lon, alt, c=edr)
sp2 = ax2.scatter(lat, alt, c=edr)
for i in range(10):
    sp3d = ax3d.scatter(np.array(dbe['longitude']), np.array(dbe['latitude']), np.array(dbe['altitude']), c=np.array(dbe['edr_peak_value']))

cb1 = plt.colorbar(sp1, ax=ax1)
cb1.set_label('Peak EDR value')
cb2 =plt.colorbar(sp2, ax=ax2)
cb2.set_label('Peak EDR value')
#cbtime = plt.colorbar(sp3d, ax=ax, fraction=0.02, pad=0.08)
#cbtime.ax.set_yticklabels([time_transform(np.array(dbt['utc_timestamp'])[i], Back=True) for i in range(len(dbt))])
#cbtime.set_label('Time of Observation')