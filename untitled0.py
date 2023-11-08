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
    
   

