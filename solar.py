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
                      solar_altitude,
                      calculate_B_0_h_new)

def import_data(dataset):
    df = pd.read_csv(dataset,sep=';',index_col='TimeStamp')
    df = df[['Cloud', 'Temp']]
    df.index = pd.to_datetime(df.index)
    return df


    # Data import and interpolation for missing data
# Load the CSV file using the import_data function
data = import_data('weather_data.csv')

# Assuming 'data' is your DataFrame
desired_length = 8290

# Truncate the DataFrame to the desired length
data = data.iloc[:desired_length]

# Round down the timestamps to the nearest hour
data.index = pd.to_datetime(data.index).ceil('H')

# Create a complete hourly index
complete_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='1H')

    # Drop duplicate labels in the index
data = data[~data.index.duplicated(keep='first')]

# Reindex the DataFrame to include all hours
data = data.reindex(complete_index)

# Linear interpolation only for missing values
data['Cloud'] = data['Cloud'].interpolate(method='linear', limit_area='inside')
data['Temp'] = data['Temp'].interpolate(method='linear', limit_area='inside')



# tilt representes inclination of the solar panel (in degress), orientation
# in degress (south=0)
tilt=13;
tilt_radians = np.radians(tilt)
orientation=0;

# Navitas coordinates
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

# Time series D_0_h
timeseries['D_0_h'] = timeseries['G_0_h'] * data['Cloud']/100 

# Calculate global horizontal irradiance on the ground
[timeseries['G_ground_h'], timeseries['solar_altitude']] = calculate_G_ground_horizontal(hours, hour_0, lon, lat, timeseries['K_t'])

# Calculate the use fraction
timeseries['F'] = calculate_diffuse_fraction(hours, hour_0, lon, lat, timeseries['K_t'])

# Calculate direct and diffuse irradiance on the horizontal surface
timeseries['B_ground_h']=[x*(1-y) for x,y in zip(timeseries['G_ground_h'], timeseries['F'])]
timeseries['D_ground_h']=[x*y for x,y in zip(timeseries['G_ground_h'], timeseries['F'])]

# Calculate extraterrestrial irradiance
timeseries['B_0_h_new'] = calculate_B_0_h_new(timeseries['G_0_h'], timeseries['D_0_h'])

# Direct radiation D(B) 
timeseries['Direct'] = timeseries['B_0_h_new'] / np.sin(np.radians(timeseries['solar_altitude']))
timeseries['Direct'].fillna(0, inplace=True)

# Diffuse radiation D(D) *isotropic*
timeseries['Diffuse'] = timeseries['D_0_h'] * (1 + np.cos(tilt_radians))/2

# Given reflectivity (ρ) and tilt angle (β) in degrees
reflectivity = 0.05
  # Assuming the tilt angle is 13 degrees

# Convert tilt angle from degrees to radians
tilt_angle = np.radians(tilt)

# Calculate albedo irradiance using the modified isotropic sky model with tilt angle
timeseries['Albedo_Irradiance'] = reflectivity * timeseries['G_0_h'] * (1 - np.cos(tilt_angle)) / 2

# Assumed module characteristics
efficiency = 0.185  # Efficiency (as a fraction)
temp_coeff_power = -0.0044  # Temperature coefficient of power (%/°C, as a fraction per °C)
STC_temperature = 25  # STC temperature in Celsius
STC_irradiance = 1000  # Irradiance at STC in W/m²

# Calculate power produced by each PV module at every hour
timeseries['Produced_Power'] = (
    255* (timeseries['Direct'] + timeseries['Diffuse'] + timeseries['Albedo_Irradiance']) / STC_irradiance *
    (1 + temp_coeff_power * (data['Temp'] - STC_temperature))
)

# Total power produced by the installation (summing all PV modules)
timeseries['Total_Produced_Power'] = 1000 * timeseries['Produced_Power']  # Assuming 1000 PV modules




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



# Extract the required data for February and June 2018
feb_data = timeseries['2018-02-01':'2018-02-08']  # First week of February
june_data = timeseries['2018-06-01':'2018-06-08']  # First week of June

# Calculate global radiation as the sum of direct, diffuse, and albedo
feb_data.loc[:, 'Global_Radiation'] = feb_data['Direct'] + feb_data['Diffuse'] + feb_data['Albedo_Irradiance']
june_data.loc[:, 'Global_Radiation'] = june_data['Direct'] + june_data['Diffuse'] + june_data['Albedo_Irradiance']

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(feb_data.index, feb_data['Global_Radiation'], label='Global Radiation (Feb)')
plt.title('Global Radiation on PV Modules (First Week of February 2018)')
plt.xlabel('Date')
plt.ylabel('Global Radiation')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(june_data.index, june_data['Global_Radiation'], label='Global Radiation (June)')
plt.title('Global Radiation on PV Modules (First Week of June 2018)')
plt.xlabel('Date')
plt.ylabel('Global Radiation')
plt.legend()

plt.tight_layout()
plt.show()



# Assuming 'timeseries' DataFrame contains the 'Total_Produced_Power' column with power estimations


# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(feb_data.index, feb_data['Total_Produced_Power'], label='Power Produced (Feb)')
plt.title('Power Produced by Installation (First Week of February 2018)')
plt.xlabel('Date')
plt.ylabel('Power Produced (W)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(june_data.index, june_data['Total_Produced_Power'], label='Power Produced (June)')
plt.title('Power Produced by Installation (First Week of June 2018)')
plt.xlabel('Date')
plt.ylabel('Power Produced (W)')
plt.legend()

plt.tight_layout()
plt.show()

#Plotting the measured production for the first week of February and the first week of June 2018.


file_path = 'New.xlsx'

production_data = pd.read_excel(file_path)

# Assuming the 'Date' column is in the format 'month/day/2018'
date_column_index = 0  # Change this index if the date column is at a different position
# Set the first column (assumed to be the 'Date' column) as the DataFrame index
production_data.set_index(production_data.columns[date_column_index], inplace=True)

# Define the relevant date range (change as needed)
start_date_Feb = '2018-02-01'
end_date_Feb = '2018-02-08'

Feb_data = production_data.loc[start_date_Feb:end_date_Feb, production_data.columns[4:29]]  # Adjust column indices accordingly

plt.figure(figsize=(10, 6))
plt.plot(Feb_data.index, Feb_data.values)
plt.title('Hourly Data for the First Week of February 2018')
plt.xlabel('Date')
plt.ylabel('Hourly Data')
plt.xticks(rotation=45)
plt.show()

start_date_June = '2018-06-01'
end_date_June = '2018-06-08'  # End date adjusted for the first week

# Extract hourly data for the first week of June
June_data = production_data.loc[start_date_June:end_date_June, production_data.columns[4:29]]

plt.figure(figsize=(10, 6))
plt.plot(June_data.index, June_data.values)
plt.title('Hourly Data for the First Week of June 2018')
plt.xlabel('Date')
plt.ylabel('Hourly Data')
plt.xticks(rotation=45)
plt.show()

