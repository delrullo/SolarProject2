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

# # Round down the timestamps to the nearest hour
# data.index = pd.to_datetime(data.index).ceil('H')

# Round the timestamps to the nearest hour
data.index = pd.to_datetime(data.index).round('H')

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
 ## Irradiance on horizontal
# Calculate extraterrestrial irradiance
timeseries['B_0_h'] = calculate_B_0_horizontal(hours, hour_0, lon, lat)  

# Clearness index is assumed to be equal to 0.7 at every hour
timeseries['K_t']=0.7*np.ones(len(hours))  

# Calculate global horizontal irradiance on the ground
[timeseries['G_ground_h'], timeseries['solar_altitude']] = calculate_G_ground_horizontal(hours, hour_0, lon, lat, timeseries['K_t'])




# Time series Diffuse irradiance on horizontal
timeseries['D_0_h'] = timeseries['G_ground_h'] * (data['Cloud']/100)

# Calculate direct irradiance on horizontal
timeseries['B_0_h_new'] = timeseries['G_ground_h'] - timeseries['D_0_h']


# Calculate the diffuse fraction
timeseries['F'] = calculate_diffuse_fraction(hours, hour_0, lon, lat, timeseries['K_t'])

# Calculate direct and diffuse irradiance on the horizontal surface
timeseries['B_ground_h']=[x*(1-y) for x,y in zip(timeseries['G_ground_h'], timeseries['F'])]
timeseries['D_ground_h']=[x*y for x,y in zip(timeseries['G_ground_h'], timeseries['F'])]

# Incident angle
timeseries['Incident angle'] = calculate_incident_angle(hours, hour_0, lon, lat,  tilt, orientation)

# Direct radiation D(B) 
timeseries['Direct'] = timeseries['B_0_h_new'] / np.sin(np.radians(timeseries['solar_altitude']))
timeseries['Direct'] = timeseries['Direct'] * np.maximum(0, np.cos(np.radians(timeseries['Incident angle'])))

timeseries['Direct'].fillna(0, inplace=True)

# Diffuse radiation D(D) *isotropic*
timeseries['Diffuse'] = timeseries['D_0_h'] * (1 + np.cos(tilt_radians))/2

# Given reflectivity (ρ) and tilt angle (β) in degrees
reflectivity = 0.05
  # Assuming the tilt angle is 13 degrees

# Convert tilt angle from degrees to radians
tilt_angle = np.radians(tilt)

# Calculate albedo irradiance using the modified isotropic sky model with tilt angle
timeseries['Albedo_Irradiance'] = reflectivity * timeseries['G_ground_h'] * (1 - np.cos(tilt_angle)) / 2

# Global irradiation G(Beta,alpha)
timeseries['Global'] = timeseries['Direct'] + timeseries['Diffuse'] + timeseries['Albedo_Irradiance']

## Plot testing
plt.figure(figsize=(12, 12))
plt.plot(timeseries['Direct']['2018-06-01 00:00':'2018-06-08 00:00'], label='Direct (June)', color='green')
plt.plot(timeseries['Diffuse']['2018-06-01 00:00':'2018-06-08 00:00'], label='Diffuse (June)', color='blue')#plt.plot(timeseries['Albedo_Irradiance']['2018-06-01 00:00':'2018-06-08 00:00'], label='Albedo_Irradiance (June)', color='red')
plt.show()
    
##

# Module characteristics
efficiency = 0.1585  # Efficiency (as a fraction)
temp_coeff_power = -0.0044  # Temperature coefficient of power (%/°C, as a fraction per °C)
STC_temperature = 25  # STC temperature in Celsius
STC_irradiance = 1000  # Irradiance at STC in W/m²
NOCT = 45 # nominal operating cell temp
no_pv = 1000    # Number of panels
area_pv = 1.640*0.922   # area of single panel [m^2]

# Temperature calc.
timeseries['T_c'] = data['Temp'] + ((NOCT-20)/800)*timeseries['Global']
# Calculate total power
timeseries['Power'] = efficiency*area_pv*STC_irradiance
# Calculate power produced in [W] by each PV module at every hour
timeseries['Produced_Power'] = (
    timeseries['Power']* (timeseries['Global']) / STC_irradiance *
    (1 + temp_coeff_power * (data['Temp'] - STC_temperature))
)

# Total power produced in [KWh] by the installation (summing all PV modules)
timeseries['Total_Produced_Power'] = (1000 * timeseries['Produced_Power']) / 1000  # Assuming 1000 PV modules



    ## Plot of Global radiation on horizontal surface G(0)
# Create a subplot with 2 rows and 1 column
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 2)
# Plotting G_0_h time series for the first week of June
plt.plot(timeseries['G_ground_h']['2018-06-01 00:00':'2018-06-08 00:00'], label='G_0_h (June)', color='blue', linewidth=3)
plt.plot(timeseries['B_0_h_new']['2018-06-01 00:00':'2018-06-08 00:00'], label='G_0_h (June)', color='green', linewidth=3)
plt.plot(timeseries['D_0_h']['2018-06-01 00:00':'2018-06-08 00:00'], label='G_0_h (June)', color='red', linewidth=3)
plt.title('Global radiation on horizontal surface (June 1st - June 7th)')
plt.ylabel(r'$\mathrm{\left[\frac{W}{m^2}\right]}$',fontsize=14)
plt.title('Global irradiation horizontal surface (June 1st - June 7th)')
plt.xticks(fontsize=12)
plt.legend(['G(0)'])

plt.grid(True)

plt.subplot(2, 1, 1)
# Plotting G_0_h time series for the first week of February
plt.plot(timeseries['G_ground_h']['2018-02-01 00:00':'2018-02-08 00:00'], label='G_0_h (February)', color='blue', linewidth=3)
plt.plot(timeseries['B_0_h_new']['2018-02-01 00:00':'2018-02-08 00:00'], label='G_0_h (February)', color='green', linewidth=3)
plt.plot(timeseries['D_0_h']['2018-02-01 00:00':'2018-02-08 00:00'], label='G_0_h (February)', color='red', linewidth=3)
plt.title('Global radiation on horizontal surface (Feb 1st - Feb 7th)')
plt.ylabel(r'$\mathrm{\left[\frac{W}{m^2}\right]}$',fontsize=14)
plt.title('Global irradiation horizontal surface (Feb 1st - Feb 7th)')
plt.xticks(fontsize=12)
plt.grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

#plot for only direct and diffuse irradiation
# Create two separate figures for February and June
fig, axs = plt.subplots(2, figsize=(12, 12))
# Plot for the first week of June
axs[0].plot(timeseries['B_0_h_new']['2018-06-01 00:00':'2018-06-08 00:00'], label='Direct Irradiation (June)', color='green', linewidth=3)
axs[0].plot(timeseries['D_0_h']['2018-06-01 00:00':'2018-06-08 00:00'], label='Diffuse Irradiation (June)', color='red', linewidth=3)
axs[0].set_title('Global Irradiation on Horizontal Surface (June 1st - June 7th)')
axs[0].set_ylabel(r'$\mathrm{G(0) \; \left[\frac{W}{m^2}\right]}$', fontsize=14)
axs[0].tick_params(axis='x', labelsize=12)
axs[0].legend()

# Plot for the first week of February
axs[1].plot(timeseries['B_0_h_new']['2018-02-01 00:00':'2018-02-08 00:00'], label='Direct Irradiation (Feb)', color='green', linewidth=3)
axs[1].plot(timeseries['D_0_h']['2018-02-01 00:00':'2018-02-08 00:00'], label='Diffuse Irradiation (Feb)', color='red', linewidth=3)
axs[1].set_title('Global Irradiation on Horizontal Surface (Feb 1st - Feb 7th)')
axs[1].set_ylabel(r'$\mathrm{G(0) \; \left[\frac{W}{m^2}\right]}$', fontsize=14)
axs[1].tick_params(axis='x', labelsize=12)
axs[1].legend()


# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
# Create a subplot with 2 rows and 1 column
plt.figure(figsize=(12, 12))

# Plotting diffuse fraction time series for the first week of June
plt.subplot(2, 1, 1)
plt.plot(timeseries['F']['2018-06-01 00:00':'2018-06-08 00:00'], label='Diffuse Fraction (June)', color='orange')
plt.title('Diffuse Fraction Time Series (June 1st - June 7th)')
plt.xticks(fontsize=12)
plt.ylabel('Diffuse Fraction')
plt.grid(True)

# Plotting diffuse fraction time series for the first week of February
plt.subplot(2, 1, 2)
plt.plot(timeseries['F']['2018-02-01 00:00':'2018-02-08 00:00'], label='Diffuse Fraction (February)', color='purple')
plt.title('Diffuse Fraction Time Series (Feb 1st - Feb 7th)')
plt.xticks(fontsize=12)
plt.ylabel('Diffuse Fraction')
plt.grid(True)

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

    ## Plotting Global Radiation
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(feb_data.index, feb_data['Global_Radiation'], label='Global Radiation (Feb)')
plt.title('Global Radiation on PV Modules (Feb 1st - Feb 7th)')
plt.ylabel(r'$\mathrm{G(\beta,\alpha) \; \left[\frac{W}{m^2}\right]}$',fontsize=14)
plt.xticks(fontsize=12)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(june_data.index, june_data['Global_Radiation'], label='Global Radiation (June)')
plt.title('Global Radiation on PV Modules (June 1st - June 7th)')
plt.ylabel(r'$\mathrm{G(\beta,\alpha) \; \left[\frac{W}{m^2}\right]}$',fontsize=14)
plt.xticks(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()



    ## Plotting Total Power Produced
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(feb_data.index, feb_data['Total_Produced_Power'], label='Power Produced (Feb)')
plt.title('Modelled Power production (Feb 1st - Feb 7th)')
plt.xticks(fontsize=12)
plt.ylabel(r'$\mathrm{P_{PV} \; [KWh]}$',fontsize=14)
plt.grid(True)
#plt.legend()

plt.subplot(2, 1, 2)
plt.plot(june_data.index, june_data['Total_Produced_Power'], label='Power Produced (June)')
plt.title('Modelled Power production (June 1st - June 7th)')
plt.xticks(fontsize=12)
plt.ylabel(r'$\mathrm{P_{PV} \; [KWh]}$',fontsize=14)
#plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#Plotting the measured production for the first week of February and the first week of June 2018.

# File path to the Excel file containing production data
file_path = 'New.xlsx'

# Load the production data from the Excel file
production_data = pd.read_excel(file_path)

## Historical data lubrication (rearranging)
# Extract relevant columns and reshape the data
hourly_data = production_data.iloc[:, 5:30].values.reshape(-1)

# Define the start date
start_date = pd.to_datetime('2018-02-01 01:00:00')

# Create a new datetime index with hourly frequency
new_index = pd.date_range(start=start_date, periods=len(hourly_data), freq='H')

# Create a new DataFrame with the desired index and 'production' column
Measured_Power = pd.DataFrame({'production': hourly_data}, index=new_index)
Measured_Power = Measured_Power.iloc[:-1]

# shifting data
Measured_Power.index = Measured_Power.index - pd.DateOffset(hours=3)
end_index = Measured_Power.index[-1] + pd.DateOffset(hours=1)  # Find the last index and add 1 hour to get the next index
new_rows = pd.DataFrame({'production': [0, 0, 0]}, index=pd.date_range(end_index, periods=3, freq='H'))
Measured_Power = pd.concat([Measured_Power, new_rows])

# Plotting data for the first week of February and the first week of June as line plots
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)

# Plotting data for the first week of February
plt.plot(Measured_Power['2018-02-01':'2018-02-08'], label='Measured Power (Feb)', color='blue')
plt.title('Measured Power production (Feb 1st - Feb 7th)')
plt.xticks(fontsize=12)
plt.ylabel(r'$\mathrm{P_{PV} \; [KWh]}$',fontsize=14)
#plt.legend()
plt.grid(True)

# Plotting data for the first week of June
plt.subplot(2, 1, 2)
plt.plot(Measured_Power['2018-06-01':'2018-06-08'], label='Measured Power (June)', color='green')
plt.title('Measured Power production (June 1st - June 7th)')
plt.ylabel(r'$\mathrm{P_{PV} \; [KWh]}$',fontsize=14)
#plt.legend()
plt.grid(True)
plt.xticks(fontsize=12)
plt.tight_layout()

plt.show()


## Comparison plots
plt.figure(figsize=(12, 12))
# February
    # Measure feb
plt.plot(Measured_Power['2018-02-01':'2018-02-08'], label='Measured Power (Feb)', color='green')
plt.plot(feb_data.index, feb_data['Total_Produced_Power'], label='Power Produced (Feb)')
plt.title('Power production (Feb 1st - Feb 7th)')
plt.xticks(fontsize=12)
plt.ylabel(r'$\mathrm{P_{PV} \; [KWh]}$',fontsize=14)
plt.legend()
plt.grid(True)
    
plt.show()

# June
plt.figure(figsize=(12, 12))

plt.plot(Measured_Power['2018-06-01':'2018-06-08'], label='Measured Power (June)', color='green')
plt.plot(june_data.index, june_data['Total_Produced_Power'], label='Power Produced (June)')
plt.title('Power production (June 1st - June 7th)')
plt.xticks(fontsize=12)
plt.ylabel(r'$\mathrm{P_{PV} \; [KWh]}$',fontsize=14)
plt.legend()
plt.grid(True)

plt.show()


## Modelled power data rearranging
start_date = pd.to_datetime('2018-02-01 01:00:00')
end_date = pd.to_datetime('2018-12-31 23:00:00')

Modelled_Power = timeseries.loc[start_date:end_date, 'Total_Produced_Power']




# Calculate the difference between modeled and measured power generation
timeseries['Error_Hourly'] = timeseries['Total_Produced_Power']['2018-02-01 01:00:00':] - Measured_Power['production']['2018-02-01 01:00:00':]

# Calculate Root Mean Square Error (RMSE) for hourly generation values
rmse_hourly = np.sqrt(np.mean(timeseries['Error_Hourly']['2018-02-01 01:00:00':] ** 2))
print(f"RMSE for hourly generation values: {rmse_hourly:.2f} KW")
# Resample data to daily, weekly, and monthly frequency
daily_errors = timeseries['Error_Hourly']['2018-02-01 01:00:00':].resample('D').sum()  # Resample to daily
weekly_errors = timeseries['Error_Hourly']['2018-02-01 01:00:00':].resample('W').sum()  # Resample to weekly
monthly_errors = timeseries['Error_Hourly']['2018-02-01 01:00:00':].resample('M').sum()  # Resample to monthly

# Calculate Root Mean Square Error (RMSE) for aggregated generation values
rmse_daily = np.sqrt(np.mean(daily_errors ** 2))
rmse_weekly = np.sqrt(np.mean(weekly_errors ** 2))
rmse_monthly = np.sqrt(np.mean(monthly_errors ** 2))

print(f"RMSE for daily generation values: {rmse_daily:.2f} KWh")
print(f"RMSE for weekly generation values: {rmse_weekly:.2f} KWh")
print(f"RMSE for monthly generation values: {rmse_monthly:.2f} KWh")


