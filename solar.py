# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from datetime import timedelta

from solarfun import (calculate_B_0_horizontal,
                      calculate_G_ground_horizontal,                      
                      calculate_diffuse_fraction,
                      calculate_incident_angle,
                      solar_altitude)

# Load the Excel file
data = pd.read_csv('weather_data.csv')




# tilt representes inclination of the solar panel (in degress), orientation
# in degress (south=0)
tilt=13;
orientation=0;
lat = 56.15886367 # latitude
lon = 10.215740203 # longitude
K_t = 0.7

year = 2018
hour_0 = datetime(year,1,1,0,0,0) - timedelta(hours=1)

hours = [datetime(year,1,1,0,0,0) 
         + timedelta(hours=i) for i in range(0,24*365)]
hours_str = [hour.strftime("%Y-%m-%d %H:%M ") for hour in hours]

timeseries = pd.DataFrame(
            index=pd.Series(
                data = hours,
                name = 'utc_time'),
            columns = pd.Series(
                data = ['B_0_h', 'K_t', 'G_ground_h', 'solar_altitude', 'F', 
                        'B_ground_h', 'D_ground_h', 'incident_angle', 
                        'B_tilted', 'D_tilted', 'R_tilted', 'G_tilted'], 
                name = 'names')
            )

# Calculate extraterrestrial irradiance
timeseries['B_0_h'] = calculate_B_0_horizontal(hours, hour_0, lon, lat)  

# Clearness index is assumed to be equal to 0.7 at every hour
timeseries['K_t']=0.7*np.ones(len(hours))  

# Timeseries G_zero
timeseries['G_0_h'] = timeseries['K_t'] * timeseries['B_0_h']

# Time series D_0
#timeseries['D_0_h'] = timeseries[G_0_H] * 

# Calculate global horizontal irradiance on the ground
[timeseries['G_ground_h'], timeseries['solar_altitude']] = calculate_G_ground_horizontal(hours, hour_0, lon, lat, timeseries['K_t'])

# Calculate diffuse fraction
timeseries['F'] = calculate_diffuse_fraction(hours, hour_0, lon, lat, timeseries['K_t'])

# Calculate direct and diffuse irradiance on the horizontal surface
timeseries['B_ground_h']=[x*(1-y) for x,y in zip(timeseries['G_ground_h'], timeseries['F'])]
timeseries['D_ground_h']=[x*y for x,y in zip(timeseries['G_ground_h'], timeseries['F'])]


# Create a subplot with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plotting G_0_h time series for the first week of June
ax1.plot(timeseries['G_0_h']['2018-06-01 00:00':'2018-06-08 00:00'], label='G_0_h (June)', color='green')
ax1.set_title('G_0_h Time Series (June 1st - June 8th)')
ax1.set_xlabel('Time')
ax1.set_ylabel('G_0_h')
ax1.legend()
ax1.grid(True)

# Plotting G_0_h time series for the first week of February
ax2.plot(timeseries['G_0_h']['2018-02-01 00:00':'2018-02-08 00:00'], label='G_0_h (February)', color='blue')
ax2.set_title('G_0_h Time Series (Feb 1st - Feb 8th)')
ax2.set_xlabel('Time')
ax2.set_ylabel('G_0_h')
ax2.legend()
ax2.grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

# Create a subplot with 2 rows and 1 column
fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))

# Plotting diffuse fraction time series for the first week of July
ax3.plot(timeseries['F']['2018-07-01 00:00':'2018-07-08 00:00'], label='Diffuse Fraction (July)', color='orange')
ax3.set_title('Diffuse Fraction Time Series (July 1st - July 8th)')
ax3.set_xlabel('Time')
ax3.set_ylabel('Diffuse Fraction')
ax3.legend()
ax3.grid(True)

# Plotting diffuse fraction time series for the first week of February
ax4.plot(timeseries['F']['2018-02-01 00:00':'2018-02-08 00:00'], label='Diffuse Fraction (February)', color='purple')
ax4.set_title('Diffuse Fraction Time Series (Feb 1st - Feb 8th)')
ax4.set_xlabel('Time')
ax4.set_ylabel('Diffuse Fraction')
ax4.legend()
ax4.grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


plt.figure(figsize=(20, 10))
gs1 = gridspec.GridSpec(2, 2)
#gs1.update(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(gs1[0,0])
ax1.plot(timeseries['G_ground_h']['2018-06-01 00:00':'2018-06-08 00:00'], 
         label='G_ground_h', color='blue')
ax1.plot(timeseries['B_ground_h']['2018-06-01 00:00':'2018-06-08 00:00'], 
         label='B_ground_h', color= 'orange')
ax1.plot(timeseries['D_ground_h']['2018-06-01 00:00':'2018-06-08 00:00'], 
         label='D_ground_h', color= 'purple')
ax1.legend(fancybox=True, shadow=True,fontsize=12, loc='best')
ax1.set_ylabel('W/m2')

