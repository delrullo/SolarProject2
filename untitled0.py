# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:28:28 2023

@author: emilb
"""

import numpy as np

def eccentricity(day): 
    """
    Calculate eccentricity.
    
    Parameters:
        day = number of the day, counted from the first day of the year 
              (1...365)    
    """
    ecc = 1 + 0.033*np.cos(360/365*day*np.pi/180)
    return ecc

def solar_altitude(latitude, declination, omega): 
    """
    Calculate solar altitude, in degrees
    
    Parameters:
        latitude = in degrees
        declination = in degrees
        omega = true solar time, expressed as angle (omega=0 when the Sun is a 
                the highest position), in degrees    
    """    
    solar_altitude = (180/np.pi)*(np.arcsin(np.sin(declination*np.pi/180)
                     *np.sin(latitude*np.pi/180) 
                     + (np.cos(declination*np.pi/180)*np.cos(latitude*np.pi/180)
                     *np.cos(omega*np.pi/180))))
    return solar_altitude


def solar_azimuth (latitude, declination, omega): 
    """
    Calculate solar azimuth, in degrees
    
    Parameters:
        latitude = in degrees
        declination = in degrees
        omega = true solar time, expressed as angle (omega=0 when the Sun is a 
                the highest position), in degrees 
    
    """
    gamma_s=solar_altitude(latitude, declination, omega)
    if latitude>0:
        sign=1
    else:
        sign=-1
    solar_azimuth = (180/np.pi)*(np.arccos(sign*(np.sin(gamma_s*np.pi/180)
                    *np.sin(latitude*np.pi/180)-np.sin(declination*np.pi/180))
                    /(np.cos(gamma_s*np.pi/180)*np.cos(latitude*np.pi/180)))) 
    return solar_azimuth 

    def ET(day):
        B = (day-81)*2*np.pi/364
        ET_ = 9.87*np.sin(2*B) - 7.53*np.cos(B)-1.5*np.sin(B)
        return ET_
    
        """
    Calculate correction for different lengths of the days in a year
    
    Parameters:
        day = number of the day, counted from the first day of the year (1...365)
    """
 


def omega(hour, day, longitude): 
    ET_=ET(day)
    omega = 15*(hour + ET_/60 - 12) + (longitude)
    return omega

def B_0_horizontal(day,solar_altitude): 
    B_0 = 1367 # [W/m2]
    B_0_horizontal=B_0*eccentricity(day)*np.sin(solar_altitude*np.pi/180)
    return B_0_horizontal
    
    """
    Calculate true solar time, expressed as angle (omega=0 when the Sun is a 
    the highest position), in degrees
    
    Parameters:
        hour 
        day = number of the day, counted from the first day of the year (1...365)
        longitude = in degrees
        def omega(hour, day, longitude): 
    """
    # TO- AO = UCT (hour is expressed in UCT)
   #reference longitude = 0 (Greenwich)
    
    """
    Calculate direct irradiance on the horizontal ground surface, in W/m2
    
    Parameters:
        day = number of the day, counted from the first day of the year (1...365)    
        solar _altitude = in degrees
    """    
  
    def calculate_B_0_horizontal(hours_year, hour_0, longitude, latitude):
        solar_altitude_ = [solar_altitude(latitude, declination((hour-hour_0).days), 
                          omega(hour.hour, (hour-hour_0).days, longitude)) 
                          for hour in hours_year]
        solar_altitude_ = [x if x>0 else 0 for x in solar_altitude_]   
        B_0_horizontal_ = [B_0_horizontal((hour-hour_0).days, solar_altitude) 
                                for hour, solar_altitude 
                                in zip(hours_year,solar_altitude_)]
        return B_0_horizontal_

        
    """
    Calculate direct irradiance on the horizontal ground surface, in W/m2
    for a series of hours 
    
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
    """ 
    

#def G_ground_horizontal(day,solar_altitude): # alternative definition, if you 
#    """                                      # don't have information on
                                              # the clearness index
#    Calculate global irradiance on the horizontal ground surface, in W/m2
#    Parameters:
#        day = number of the day, counted from the first day of the year (1...365)    
#        solar _altitude = in degrees
#    """
#    
#    if np.sin(solar_altitude*np.pi/180) < np.sin(1*np.pi/180):
#        G_ground_horizontal=0
#    else:
#        B_0= 1367 # [W/m2]
#        G_ground_horizontal=B_0*(0.74**((1/(np.sin(solar_altitude*np.pi/180)))**0.678))*np.sin(solar_altitude*np.pi/180)
#        #*eccentricity(day)
#        #it can be multiplied by included 10% diffuse radiation
#    return G_ground_horizontal

def G_ground_horizontal(day, solar_altitude, clearness_index):
    G_ground_horizontal = clearness_index*B_0_horizontal(day,solar_altitude)    
    return G_ground_horizontal
    """
    Calculate global irradiance on the horizontal ground surface, in W/m2

    Parameters:
        day = number of the day, counted from the first day of the year (1...365)    
        solar _altitude = in degrees
        clearness_index
    """
    

    
def calculate_G_ground_horizontal(hours_year, hour_0, longitude, latitude, clearness_index_):
    solar_altitude_ = [solar_altitude(latitude, declination((hour-hour_0).days), 
                       omega(hour.hour, (hour-hour_0).days, longitude)) 
                       for hour in hours_year]
    solar_altitude_ = [x if x>0 else 0. for x in solar_altitude_]

    G_ground_horizontal_ = [G_ground_horizontal((hour-hour_0).days, 
                            solar_altitude, clearness_index) 
                            for hour, solar_altitude, clearness_index 
                            in zip(hours_year,solar_altitude_, clearness_index_)]
    return G_ground_horizontal_ , solar_altitude_ 
    """
    Calculate global irradiance on the horizontal ground surface, in W/m2
    for a series of hours 
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
        clearnes_index_ = list of clearness indices for the list of hours
    """ 
    

    
   

