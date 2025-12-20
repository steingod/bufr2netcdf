#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from unittest.mock import NonCallableMock
import pandas as pd
import xarray as xr
import yaml
import time
from itertools import dropwhile
import sys 
import os
import subprocess
from eccodes import *
import requests
import argparse
from io import StringIO
import logging
from logging.handlers import TimedRotatingFileHandler
import matplotlib.pyplot as plt
import numpy as np
import traceback
import operator
import re
from dateutil.relativedelta import relativedelta
from funcs.get_keywords import *
from funcs.code_tables import *
from funcs.useful_functions import *
import netCDF4
import glob
from datetime import datetime, timedelta, date
from calendar import monthrange, month_name

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c","--cfg",dest="cfgfile",
            help="Configuration file", required=True)
    parser.add_argument("-s","--startday",dest="startday",
            help="Start day in the form YYYY-MM-DD", required=False)
    parser.add_argument("-e","--endday",dest="endday",
            help="End day in the form YYYY-MM-DD", required=False)
    parser.add_argument("-t","--type",dest="stationtype",
        help="Choose between block, state, wigos and ship", required=True)
    parser.add_argument("-i", "--init", dest="initialize",
            help="Download all data", required=False, action="store_true")
    parser.add_argument("-u", "--update", dest="update",
            help="Adds the latest bufr-data to the netcdf-file NOT NEEDED", required=False, action="store_true")
    parser.add_argument("-a", "--all", dest="all_stations",
            help="To download/upload data from all stations", required=False, action="store_true")
    parser.add_argument("-st", "--station", nargs='*', default=[], dest="spec_station",
            help='To select specific stations. Must be in the from "st1 st2 .. stn". Must not be separated by comma', required=False)
    args = parser.parse_args()
    
    if args.startday is None:
        pass
    else:
        try:
            datetime.strptime(args.startday,'%Y-%m-%d')
        except ValueError:
            raise ValueError
        
    if args.endday is None:
            pass
    else:
        try:
            datetime.strptime(args.endday,'%Y-%m-%d')
        except ValueError:
            raise ValueError
        
    if args.cfgfile is None:
        parser.print_help()
        parser.exit()
        
    if args.stationtype is None:
        parser.print_help()
        parser.exit
    return args

def parse_cfg(cfgfile):

    with open(cfgfile, 'r') as ymlfile:
        cfgstr = yaml.full_load(ymlfile)
    return cfgstr

def get_files_specified_dates(desired_path):
    
    get_args = parse_arguments()
    startday = get_args.startday
    endday = get_args.endday
    path = parse_cfg(get_args.cfgfile)['station_info']['path']

    # open up path to files and sort the files such that the date is in chronological order
    files = []
    for file in os.listdir(desired_path):
        if file.endswith('.bufr'):
            files.append(datetime.strptime(file[5:-5], "%Y%m%d%H"))
    
    files = sorted(files)
    sorted_files = []
    for i in files:
        sorted_files.append(path + "/syno_" + datetime.strftime(i, "%Y%m%d%H") + '.bufr')
    
    # get startday from parser
    start = parse_arguments().startday
    end = parse_arguments().endday
    
    start = start.replace('-','')
    end = end.replace('-','')
    
    startday = []       
    endday = [] 
    for i in sorted_files:
        if start in i:
           startday.append(i) 
        if end in i:
            endday.append(i)
            
    if len(startday) == 0:
        print('your startdate is out of range')
        sys.exit()
    if len(endday) == 0:
        print('your endday is out of range')
        sys.exit()
        
    startpoint_index = sorted_files.index(startday[0])
    endpoint_index = sorted_files.index(endday[-1]) + 1
    
    return sorted_files[startpoint_index:endpoint_index]

def get_files_initialize(desired_path):
    """
    gets a list of the BUFR files that we wish to look at
    """

    files = []
    for file in os.listdir(desired_path):
        if file.endswith('.bufr'):
            # Create the filepath of particular file
            file_path = ('{}/{}'.format(desired_path,file))
            files.append(file_path)
    return files



def bufr_2_json(file):
    """
    open bufr file, convert to json (using "-j s") and load it with json.loads

    FIXME it may look like this is done too often and rather should be done in a more efficient manner... e.g. dumping to file and rereading those files if necessary. then it could be done in multi-processing as well.
    """

    # TODO add some error checking...
    ##json_file = json.loads(subprocess.check_output("bufr_dump -j f -f {}".format(file), shell=True, stderr=subprocess.STDOUT))
    mydump = subprocess.run("bufr_dump -j f -f {}".format(file), shell=True, check=False, capture_output=True)
    json_file = json.loads(mydump.stdout)
    #print(json.dumps(json_file, indent=2))
    #sys.exit()
   
    # sorting so that each subset is a list of dictionary
    count = 0
    sorted_messages = []
    for message in json_file['messages']:
        """
        print('>>>> new message ', count)
        print(json.dumps(message, indent=2))
        """
        if message['key'] == 'subsetNumber':
            count += 1
            sorted_messages.append([])
        else:
            try:
                sorted_messages[count-1].append({'key': message['key'], 'value': message['value'],
                                   'code': message['code'], 'units': message['units']})
            except:
                print('this did not work')
    
    # sorting so that height of measurement equipment is a variable attribute instead of variable
    final_message = []        
    for msg in sorted_messages:
        height_measurements = ['007006', '007030', '007031', '007032', '007033'] # might have to add more codes here, check: https://confluence.ecmwf.int/display/ECC/WMO%3D30+element+table
        count_height = 0
        storing_height = []
        variables_with_height = []
        for i in msg:
        
            # if variable is a height of sensor etc, check if it is none. If not none, then add it as an attribute
            # for the consecutive variables until next height of sensor.
            if i['code'] in height_measurements and i['value'] != None:
                count_height += 1
                storing_height.append(i)
            if i['code'] in height_measurements and i['value'] == None:
                count_height += 1
                storing_height.append(None)

            if count_height == 0:
                variables_with_height.append(i)
            if count_height != 0 and i['code'] not in height_measurements:
                if storing_height[count_height-1] != None and i['key'] != 'timePeriod':
                    i['height'] = storing_height[count_height-1]
                variables_with_height.append(i)
                
        final_message.append(variables_with_height)
    
        
    finale= []
    for msg in final_message:
        time_code = ['004024', '004025']
        count_time = 0
        variables_with_time = []
        storing_time = []
        for i in msg:
            if i['code'] in time_code and i['value'] != None:
                count_time += 1
                storing_time.append(i)
            if i['code'] in time_code and i['value'] == None:
                count_time += 1
                storing_time.append(None)

            if count_time == 0:
                variables_with_time.append(i)
            if count_time != 0 and i['code'] not in time_code:
                if storing_time[count_time-1] != None:
                    i['time'] = storing_time[count_time-1]
                variables_with_time.append(i)
        finale.append(variables_with_time) 
    return finale

def return_list_of_stations(get_files):
    """
    Get station identifiers from the files available.
    How identifiers are set up depends on the type of station.
    """
    cfg = parse_cfg(parse_arguments().cfgfile)
    stationtype = parse_arguments().stationtype
    stations = []
    if stationtype == 'block':
        for one_file in get_files:
            simple_file = bufr_2_json(one_file)
            for station in simple_file:
                if station[0]['key'] == 'blockNumber' and station[1]['key'] == 'stationNumber':
                    station_num =  str(station[0]['value']).zfill(2) + str(station[1]['value']).zfill(3)
                    if station_num not in stations:
                        stations.append(station_num)
    if stationtype == 'wigos':
        for one_file in get_files:
            simple_file = bufr_2_json(one_file)
            for station in simple_file:
                if station[0]['key'] == 'wigosIdentifierSeries' and station[1]['key'] == 'wigosIssuerOfIdentifier' and station[2]['key'] == 'wigosIssueNumber' and station[3]['key'] == 'wigosLocalIdentifierCharacter':
                    station_num =  str(station[0]['value']) + '-' + str(station[1]['value']) + '-' + str(station[2]['value']) + '-' + str(station[3]['value'])
                    if station_num not in stations:
                        stations.append(station_num)
    if stationtype == 'state':
        for one_file in get_files:
            simple_file = bufr_2_json(one_file)
            for station in simple_file:
                if station[0]['key'] == 'stateIdentifier' and station[1]['key'] == 'nationalStationNumber':
                    station_num = str(station[0]['value']) + '-' + str(station[1]['value'])
                    if station_num not in stations:
                        stations.append(station_num)
    if stationtype == 'ship':
        for one_file in get_files:
            simple_file = bufr_2_json(one_file)
            for station in simple_file:
                if station[0]['key'] == 'shipOrMobileLandStationIdentifier':
                    station_num = str(station[0]['value'])
                    if station_num not in stations:
                        stations.append(station_num)
    return stations

def sorting_hat(get_files, stations = 1):
    """
    Sorting BUFR files according to some rules (not sure yet)
    """
    cfg = parse_cfg(parse_arguments().cfgfile)
    
    if parse_arguments().spec_station:
        stations = parse_arguments().spec_station
    elif not parse_arguments().spec_station:
        stations = return_list_of_stations(get_files)
    print(stations)
    #sys.exit()
    stations_dict = {i : [] for i in stations}
    
    for one_file in get_files:
        simple_file = bufr_2_json(one_file)
        for station in simple_file:
            if parse_arguments().stationtype == 'block':
                if str(station[0]['value']).zfill(2) + str(station[1]['value']).zfill(3) in stations:
                    stations_dict['{}'.format(str(station[0]['value']).zfill(2) + str(station[1]['value']).zfill(3))].append(station)
            elif parse_arguments().stationtype == 'wigos':
                if str(station[0]['value']) + '-' + str(station[1]['value']) + '-' + str(station[2]['value']) + '-' + str(station[3]['value']) in stations:
                    stations_dict['{}'.format(str(station[0]['value']) + '-' + str(station[1]['value']) + '-' + str(station[2]['value']) + '-' + str(station[3]['value']))].append(station)
            elif parse_arguments().stationtype == 'state':
                if str(station[0]['value']) + '-' + str(station[1]['value']) in stations:
                    stations_dict['{}'.format(str(station[0]['value']) + '-' + str(station[1]['value']))].append(station)
            elif parse_arguments().stationtype == 'ship':
                if str(station[0]['value']) in stations:
                    stations_dict['{}'.format(str(station[0]['value']))].append(station)
    
    """
    seems incorrect... commented out
    print('bufr successfully converted to json')
    """
    print('sorting hat completed')
    
    #print(stations_dict)
    #sys.exit()
    return stations_dict                        
                    

def block_wigos_state(msg):
    
    # storing data 
    gathered_df = []
    gathered_df_units = []
    gathered_df_height = []
    gathered_df_time = []
    
    #the following section is still messy, could probably be cleaned better
    #currently saving the values, units, height measurement and time resolution to each df and matching it up later
    #this is mostly because I find it easy to compare dfs
    
    for i in msg:
        
        height_copy = copy_dict(i,'key','height')
        dict_copy = filter_section(copy_dict(i,'key','value','units','code'))
        time_copy = copy_dict(i,'key','time')
        
        # Create data frame from bufr json dump
        df = pd.DataFrame(dict_copy)
        #print(df)
     
        #save a table with units for later use
        units_df = df
        #print(units_df)
        units_df = units_df.drop(columns=['value'])
        #print(units_df)
        units_df = rename_duplicated_columns(units_df) # Not sure if needed
        #print(units_df)
        """
        Removed Øystein Godøy, METNO/FOU, 2025-12-21 
        cols = pd.io.parsers.base_parser.ParserBase({'names':units_df['key'], 'usecols':None})._maybe_dedup_names(units_df['key'])
        units_df['key'] = cols # will add a ".1",".2" etc for each double name
        """
        
        #save a table with height of measurement for later use
        height_df = pd.DataFrame(height_copy)
        try:
            height_df['height_numb'] = [str(val.get('value')) + ' ' + str(val.get('units')) for val in height_df.height]
            height_df['height_type'] = [str(val.get('key')) for val in height_df.height]
            height_df = height_df.drop(columns=['height'])
            height_df = rename_duplicated_columns(height_df)
            """
            Removed Øystein Godøy, METNO/FOU, 2025-12-21 
            cols = pd.io.parsers.base_parser.ParserBase({'names':height_df['key'], 'usecols':None})._maybe_dedup_names(height_df['key'])
            height_df['key'] = cols # will add a ".1",".2" etc for each double name
            """
            height_df = height_df.reset_index()
        except:
            height_df = height_df
            
        #save a table with time of measurement for later use
        time_df = pd.DataFrame(time_copy)
        try:
            time_df['time_duration'] = [str(abs(val.get('value'))) + ' ' + str(val.get('units')) for val in time_df.time]
            time_df = time_df.drop(columns=['time'])
            time_df = rename_duplicated_columns(time_df)
            """
            Removed Øystein Godøy, METNO/FOU, 2025-12-21 
            cols = pd.io.parsers.base_parser.ParserBase({'names':time_df['key'], 'usecols':None})._maybe_dedup_names(time_df['key'])
            time_df['key'] = cols # will add a ".1",".2" etc for each double name
            """
            time_df = time_df.reset_index()
        except:
            time_df = time_df
        
        
        #saving main table containing value of variables
        df = df.transpose().reset_index()
        df.columns = df.iloc[0]
        df = df[1:-2]
        df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])#.dt.strftime('%Y-%m-%d %H:%M')
        df = df.set_index('time')

        #some columnnames repeat itself, creating problems when changing to xarray. Fix with:
        df = rename_duplicated_columns(df)
        """
        Removed Øystein Godøy, METNO/FOU, 2025-12-21 
        cols = pd.io.parsers.base_parser.ParserBase({'names':df.columns, 'usecols':None})._maybe_dedup_names(df.columns)
        df.columns = cols # will add a ".1",".2" etc for each double name
        """
        df = df.drop(columns=['key', 'year', 'month', 'day', 'hour', 'minute'])
        gathered_df.append(df)
        gathered_df_units.append(units_df)
        gathered_df_height.append(height_df)
        gathered_df_time.append(time_df)
    

     # making sure that we managed to gather data, and it takes into account if we're only looking at one dataset too

    if len(gathered_df) == 0:
        print('There is no data to collect')
        sys.exit()
        
    if len(gathered_df_units) > 1:
        units_df = units(gathered_df_units).reset_index()
    else:
        units_df = gathered_df_units[0]
    
    if len(gathered_df_height) > 1:
        height_df = height(gathered_df_height).reset_index()
    else:
        height_df = gathered_df_height[0]
        
    if len(gathered_df_time) > 1:
        time_df = times(gathered_df_time).reset_index()
    else:
        time_df = gathered_df_time[0]
    
    if len(gathered_df) > 1:
        main_df = pd.concat(gathered_df)
    else:
        main_df = gathered_df[0]
    
    if height_df.empty:
        height_df['key'] = pd.DataFrame({'key': ['something','to','make', 'it' ,'pass']})
    if time_df.empty:
        time_df['key'] = pd.DataFrame({'key': ['something','to','make', 'it' ,'pass']})
    if units_df.empty:
        units_df= pd.DataFrame({'key': ['something','to','make', 'it' ,'pass']})
    
    
    #converting over to numeric.
    
    for column in main_df.columns:
        try:
            main_df[column] = pd.to_numeric(main_df[column])
        except:
            main_df[column] = main_df[column]
    
    #over to dataset
    main_ds = main_df.to_xarray()
    main_ds = main_ds.fillna(-9999)

    # some variables only need the first value as they are not dependent on time
    vars_one_val = ['blockNumber', 'stationNumber','latitude',
                    'longitude', 'heightOfStation', 'wigosIdentifierSeries', 'wigosIssuerOfIdentifier',
                    'wigosIssueNumber','wigosLocalIdentifierCharacter', 'nationalStationNumber', 
                    'stationOrSiteName','stationType', 'stateIdentifier']
    for i in main_ds.keys():
        if i in vars_one_val:
            main_ds[i] = main_ds[i].isel(time=0)
    
    # need to get the set of keywords
    to_get_keywords = []
    
    # VARIABLE ATTRIBUTES
    for variable in main_ds.keys():
        
        
        var = variable
        if variable[-2] == '.':
            var = variable[:-2]
        change_upper_case = re.sub(r'(?<!^)(?=\d{2})', '_',re.sub('(?<!^)(?=[A-Z])', '_', var)).lower()
        
        
        
        # checking for standardname
        standardname_check = cf_match(change_upper_case)
        if standardname_check != 'fail':
            main_ds[variable].attrs['standard_name'] = standardname_check
        manual_stdname = {'windDirection': 'wind_from_direction', 'dewpointTemperature': 'dew_point_temperature',
                          'heightOfBaseOfCloud':'cloud_base_altitude', 
                          'characteristicOfPressureTendency':'tendency_of_air_pressure',
                          'horizontalVisibility':'visibility_in_air', 
                          'totalPrecipitationOrTotalWaterEquivalent': 'precipitation_amount',
                          'totalSnowDepth':'lwe_thickness_of_snowfall_amount'}
        if var in manual_stdname.keys():
            main_ds[variable].attrs['standard_name'] = manual_stdname[var]

            
            
        # adding long name. if variable has a height attribute, add it in the long name. if variable has a time attribute, add it in the long name.
        long_name = re.sub('_',' ', change_upper_case)
        if variable in height_df.key.values.tolist() and has_numbers(long_name) == False:
            main_ds[variable].attrs['long_name'] = (long_name + ' measured at ' + 
                                                    height_df.loc[height_df['key'] == variable, 'height_numb'].iloc[0])
        elif variable in time_df.key.values.tolist() and has_numbers(long_name) == False:
            main_ds[variable].attrs['long_name'] = (long_name + ' with time duration of ' + 
                                                    time_df.loc[time_df['key'] == variable, 'time_duration'].iloc[0])
        elif variable in height_df.key.values.tolist() and variable in time_df.key.values.tolist() and has_numbers(long_name) == False:
                        main_ds[variable].attrs['long_name'] = (long_name + ' measured at ' +
                                                                height_df.loc[height_df['key'] == variable, 'height_numb'].iloc[0] + 
                                                                ' with time duration of ' + 
                                                                time_df.loc[time_df['key'] == variable, 'time_duration'].iloc[0])
        else:
            main_ds[variable].attrs['long_name'] = long_name

            
            
        # adding units, if units is a CODE TABLE or FLAG TABLE, it will overrule the previous set attributes to set new ones
        if var in list(units_df['key']):
            main_ds[variable].attrs['units'] = units_df.loc[units_df['key'] == var, 'units'].iloc[0]
        else:
            continue
        
        
        
        if main_ds[variable].attrs['units'] == 'deg' and variable not in ['longitude', 'latitude']:
            main_ds[variable].attrs['units'] = 'degrees'
        elif variable == 'latitude' and main_ds[variable].attrs['units'] == 'deg':
            main_ds[variable].attrs['units'] = 'degrees_north'
        elif variable == 'longitude' and main_ds[variable].attrs['units'] == 'deg':
            main_ds[variable].attrs['units'] = 'degrees_east'
        elif main_ds[variable].attrs['units'] == 'Numeric':
            main_ds[variable].attrs['units'] = '1'
        if main_ds[variable].attrs['units'] == 'CODE TABLE':
            del main_ds[variable].attrs['units']
            main_ds[variable].attrs['long_name'] = main_ds[variable].attrs['long_name'] + ' according to WMO code table ' + units_df.loc[units_df['key'] == variable, 'code'].iloc[0]  
            main_ds[variable].attrs['units'] = '1'
        if main_ds[variable].attrs['units'] == 'FLAG TABLE':
            main_ds[variable].attrs['long_name'] = main_ds[variable].attrs['long_name'] + ' according to WMO flag table ' + units_df.loc[units_df['key'] == variable, 'code'].iloc[0]
            main_ds[variable].attrs['units'] = '1'
            
            
            

        # adding coverage_content_type
        thematicClassification = ['blockNumber', 'stationNumber', 'stationType', 'wigosIdentifierSeries', 'wigosIssuerOfIdentifier', 'wigosIssueNumber', 'wigosLocalIdentifierCharacter', 'stationOrSiteName', 'stationType', 'stateIdentifier']
        
        if variable in thematicClassification:
            main_ds[variable].attrs['coverage_content_type'] = 'thematicClassification'
        else:
            main_ds[variable].attrs['coverage_content_type'] = 'physicalMeasurement'
        to_get_keywords.append(change_upper_case)

        
        
       
        if variable[-2] == '.':
            new_name = variable.replace('.', '')
            main_ds[new_name] = main_ds[variable]
            main_ds = main_ds.drop([variable])
            
        to_get_keywords.append(change_upper_case)
        
        # must rename if variable name starts with a digit
        timeunits = ['hour', 'second', 'minute', 'year', 'month', 'day']
        if change_upper_case[0].isdigit() == True and change_upper_case.split('_')[1].lower() in timeunits:
            fixing_varname = re.sub( r"([A-Z])", r" \1", variable).split()
            fixing_varname = fixing_varname[2:] + fixing_varname[:2] + ['s']
            fixing_varname = ''.join(fixing_varname)
            fixing_varname = fixing_varname[0].lower() + fixing_varname[1:]
            main_ds[fixing_varname] = main_ds[variable]
            main_ds = main_ds.drop([variable])
            
           
    main_ds = main_ds.assign_coords({'latitude': main_ds['latitude'].values, 'longitude': main_ds['longitude'].values})

    main_ds['longitude'].attrs['long_name'] = 'longitude'
    main_ds['longitude'].attrs['standard_name'] = 'longitude'
    main_ds['longitude'].attrs['units'] = 'degrees_east'
    main_ds['longitude'].attrs['coverage_content_type'] = 'coordinate'

    main_ds['latitude'].attrs['long_name'] = 'latitude'
    main_ds['latitude'].attrs['standard_name'] = 'latitude'
    main_ds['latitude'].attrs['units'] = 'degrees_north'
    main_ds['latitude'].attrs['coverage_content_type'] = 'coordinate'

    #print(main_ds)
    #sys.exit()
    
    ################# REQUIRED GLOBAL ATTRIBUTES ###################
    # this probably might have to be modified
    
    keywords, keywords_voc = get_keywords(to_get_keywords)
    cfg = parse_cfg(parse_arguments().cfgfile)
    main_ds.attrs['featureType'] = 'timeSeries'
    main_ds.attrs['naming_authority'] = 'World Meteorological Organization (WMO)'
    main_ds.attrs['source'] = cfg['output']['source']
    main_ds.attrs['summary'] = cfg['output']['abstract']
    
    main_ds.attrs['date_created'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
    main_ds.attrs['geospatial_lat_min'] = '{:.3f}'.format(main_ds['latitude'].values.min())
    main_ds.attrs['geospatial_lat_max'] = '{:.3f}'.format(main_ds['latitude'].values.max())
    main_ds.attrs['geospatial_lon_min'] = '{:.3f}'.format(main_ds['longitude'].values.min())
    main_ds.attrs['geospatial_lon_max'] = '{:.3f}'.format(main_ds['longitude'].values.max())
    main_ds.attrs['time_coverage_start'] = main_ds['time'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S') # note that the datetime is changed to microsecond precision from nanosecon precision
    main_ds.attrs['time_coverage_end'] = main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')
    
    duration_years = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).years)
    duration_months = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).months)
    duration_days = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).days)
    duration_hours = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).hours)
    duration_minutes = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).minutes)
    duration_seconds = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).seconds)
    main_ds.attrs['time_coverage_duration'] = ('P' + duration_years + 'Y' + duration_months +
                                               'M' + duration_days + 'DT' + duration_hours + 
                                               'H' + duration_minutes + 'M' + duration_seconds + 'S')    
    
    main_ds.attrs['keywords'] = keywords
    main_ds.attrs['keywords_vocabulary'] = keywords_voc
    main_ds.attrs['standard_name_vocabulary'] = 'CF Standard Name V79'
    main_ds.attrs['Conventions'] = 'ACDD-1.3, CF-1.6'
    main_ds.attrs['creator_type'] = cfg['author']['creator_type']
    main_ds.attrs['institution'] = cfg['author']['PrincipalInvestigatorOrganisation']
    main_ds.attrs['creator_name'] = cfg['author']['PrincipalInvestigator']
    main_ds.attrs['creator_email'] = cfg['author']['PrincipalInvestigatorEmail']
    main_ds.attrs['creator_url'] = cfg['author']['PrincipalInvestigatorOrganisationURL']
    main_ds.attrs['publisher_name'] = cfg['author']['Publisher']
    main_ds.attrs['publisher_email'] = cfg['author']['PublisherEmail']
    main_ds.attrs['publisher_url'] = cfg['author']['PublisherURL']
    main_ds.attrs['project'] = cfg['author']['Project']
    main_ds.attrs['license'] = cfg['author']['License']
    
    return main_ds



def shipOrMobileLandStationIdentifier(msg):
    gathered_df = []
    gathered_df_units = []
    gathered_df_height = []
    gathered_df_time = []
    
    for i in msg:

        height_copy = copy_dict(i,'key','height')
        dict_copy = copy_dict(i, 'key','value', 'units', 'code')
        time_copy = copy_dict(i,'key','time')
        
        df = pd.DataFrame(dict_copy)

        #save a table with units for later use
        units_df = df
        units_df = units_df.drop(columns=['value'])
        # Check for same as above
        cols = pd.io.parsers.base_parser.ParserBase({'names':units_df['key'], 'usecols':None})._maybe_dedup_names(units_df['key'])
        units_df['key'] = cols # will add a ".1",".2" etc for each double name

        #save a table with height of measurement for later use
        height_df = pd.DataFrame(height_copy)
        try:
            height_df['height_numb'] = [str(val.get('value')) + ' ' + str(val.get('units')) for val in height_df.height]
            height_df['height_type'] = [str(val.get('key')) for val in height_df.height]
            height_df = height_df.drop(columns=['height'])
            cols = pd.io.parsers.base_parser.ParserBase({'names':height_df['key'], 'usecols':None})._maybe_dedup_names(height_df['key'])
            height_df['key'] = cols # will add a ".1",".2" etc for each double name
            height_df = height_df.reset_index()
        except:
            height_df = height_df  

        #save a table with time of measurement for later use
        time_df = pd.DataFrame(time_copy)
        try:
            time_df['time_duration'] = [str(abs(val.get('value'))) + ' ' + str(val.get('units')) for val in time_df.time]
            time_df = time_df.drop(columns=['time'])
            cols = pd.io.parsers.base_parser.ParserBase({'names':time_df['key'], 'usecols':None})._maybe_dedup_names(time_df['key'])
            time_df['key'] = cols # will add a ".1",".2" etc for each double name
            time_df = time_df.reset_index()
        except:
            time_df = time_df
        
        df = df.transpose().reset_index()
        df.columns = df.iloc[0]
        df = df[1:-2]
        df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])#.dt.strftime('%Y-%m-%d %H:%M')
        df = df.set_index(['time'])

        #some columnnames repeat itself, creating problems when changing to xarray. Fix with:
        cols = pd.io.parsers.base_parser.ParserBase({'names':df.columns, 'usecols':None})._maybe_dedup_names(df.columns)
        df.columns = cols # will add a ".1",".2" etc for each double name
        df = df.drop(columns=['key', 'year', 'month', 'day', 'hour', 'minute'])
    
        gathered_df.append(df)
        gathered_df_units.append(units_df)
        gathered_df_height.append(height_df)
        gathered_df_time.append(time_df)
    #print(gathered_df)

    if len(gathered_df) == 0:
        print('There is no data to collect')
        sys.exit()
        
    if len(gathered_df_units) > 1:
        units_df = units(gathered_df_units).reset_index()
    else:
        units_df = gathered_df_units[0]
    
    if len(gathered_df_height) > 1:
        height_df = height(gathered_df_height).reset_index()
    else:
        height_df = gathered_df_height[0]
        
    if len(gathered_df_time) > 1:
        time_df = times(gathered_df_time).reset_index()
    else:
        time_df = gathered_df_time[0]
    
    if len(gathered_df) > 1:
        main_df = pd.concat(gathered_df)
    else:
        main_df = gathered_df[0]
        
    # need to check if the dfs are empty
    if height_df.empty:
        height_df = pd.DataFrame({'key': ['something','to','make', 'it' ,'pass']})
    if time_df.empty:
        time_df= pd.DataFrame({'key': ['something','to','make', 'it' ,'pass']})
    if units_df.empty:
        units_df= pd.DataFrame({'key': ['something','to','make', 'it' ,'pass']})

            
    # converting to numeric
    for column in main_df.columns:
        try:
            main_df[column] = pd.to_numeric(main_df[column])
        except:
            main_df[column] = main_df[column]
    
    #over to dataset
    main_ds = main_df.to_xarray()

    #save for later   
    main_ds, shipOrMobileLandStationIdentifier = check_ds(main_ds, 'shipOrMobileLandStationIdentifier')
    main_ds, stationOrSiteName = check_ds(main_ds, 'stationOrSiteName')
    main_ds, stationType = check_ds(main_ds, 'stationType')
    
    to_get_keywords = []
    
    # VARIABLE ATTRIBUTES
    for variable in main_ds.keys():
        var = variable
        if variable[-2] == '.':
            var = variable[:-2]
        change_upper_case = re.sub(r'(?<!^)(?=\d{2})', '_',re.sub('(?<!^)(?=[A-Z])', '_', var)).lower()
        
        # checking for standardname
        standardname_check = cf_match(change_upper_case)
        if standardname_check != 'fail':
            main_ds[variable].attrs['standard_name'] = standardname_check
        
        # adding long name. if variable has a height attribute, add it in the long name. if variable has a time attribute, add it in the long name.
        long_name = re.sub('_',' ', change_upper_case)
   
        if variable in height_df.key.values.tolist() and has_numbers(long_name) == False:
            main_ds[variable].attrs['long_name'] = (long_name + ' measured at ' + 
                                                    height_df.loc[height_df['key'] == variable, 'height_numb'].iloc[0])
        elif variable in time_df.key.values.tolist() and has_numbers(long_name) == False:
            main_ds[variable].attrs['long_name'] = (long_name + ' with time duration of ' + 
                                                    time_df.loc[time_df['key'] == variable, 'time_duration'].iloc[0])
        elif variable in height_df.key.values.tolist() and variable in time_df.key.values.tolist() and has_numbers(long_name) == False:
                        main_ds[variable].attrs['long_name'] = (long_name + ' measured at ' +
                                                                height_df.loc[height_df['key'] == variable, 'height_numb'].iloc[0] + 
                                                                ' with time duration of ' + 
                                                                time_df.loc[time_df['key'] == variable, 'time_duration'].iloc[0])
        else:
            main_ds[variable].attrs['long_name'] = long_name

        # adding units, if units is a CODE TABLE or FLAG TABLE, it will overrule the previous set attributes to set new ones
        main_ds[variable].attrs['units'] = units_df.loc[units_df['key'] == variable, 'units'].iloc[0]
        if main_ds[variable].attrs['units'] == 'deg':
            main_ds[variable].attrs['units'] = 'degrees'
        if main_ds[variable].attrs['units'] == 'Numeric':
            main_ds[variable].attrs['units'] = '1'
        if main_ds[variable].attrs['units'] == 'CODE TABLE':
            del main_ds[variable].attrs['units']
            main_ds[variable].attrs['long_name'] = main_ds[variable].attrs['long_name'] + ' according to WMO code table ' + units_df.loc[units_df['key'] == variable, 'code'].iloc[0]  
            main_ds[variable].attrs['units'] = '1'
        elif main_ds[variable].attrs['units'] == 'FLAG TABLE':
            main_ds[variable].attrs['long_name'] = main_ds[variable].attrs['long_name'] + ' according to WMO flag table ' + units_df.loc[units_df['key'] == variable, 'code'].iloc[0]
            main_ds[variable].attrs['units'] = '1'

        # adding coverage_content_type
        thematicClassification = ['blockNumber', 'stationNumber', 'stationType']
        if variable in thematicClassification:
            main_ds[variable].attrs['coverage_content_type'] = 'thematicClassification'
        else:
            main_ds[variable].attrs['coverage_content_type'] = 'physicalMeasurement'
        to_get_keywords.append(change_upper_case)

        # must rename if variable name starts with a digit
        timeunits = ['hour', 'second', 'minute', 'year', 'month', 'day']
        if change_upper_case[0].isdigit() == True and change_upper_case.split('_')[1].lower() in timeunits:
            fixing_varname = re.sub( r"([A-Z])", r" \1", variable).split()
            fixing_varname = fixing_varname[2:] + fixing_varname[:2] + ['s']
            fixing_varname = ''.join(fixing_varname)
            fixing_varname = fixing_varname[0].lower() + fixing_varname[1:]
            main_ds[fixing_varname] = main_ds[variable]
            main_ds = main_ds.drop([variable])
        
        # must also rename the ones that end with ".1"
        if variable[-2] == '.':
            new_name = variable.replace('.', '')
            main_ds[new_name] = main_ds[variable]
            main_ds = main_ds.drop([variable])

    main_ds['time'].attrs['long_name'] = 'time'
    main_ds['time'].attrs['standard_name'] = 'time'
    main_ds['time'].attrs['coverage_content_type'] = 'referenceInformation'   

    main_ds = main_ds.assign_coords({'latitude': main_ds['latitude'].values, 'longitude': main_ds['longitude'].values})

    main_ds['longitude'].attrs['long_name'] = 'longitude'
    main_ds['longitude'].attrs['standard_name'] = 'longitude'
    main_ds['longitude'].attrs['units'] = 'degrees_east'
    main_ds['longitude'].attrs['coverage_content_type'] = 'coordinate'

    main_ds['latitude'].attrs['long_name'] = 'latitude'
    main_ds['latitude'].attrs['standard_name'] = 'latitude'
    main_ds['latitude'].attrs['units'] = 'degrees_north'
    main_ds['latitude'].attrs['coverage_content_type'] = 'coordinate'

    keywords, keywords_voc = get_keywords(to_get_keywords)
    cfg = parse_cfg(parse_arguments().cfgfile)
    
    ################# REQUIRED GLOBAL ATTRIBUTES ###################
    # this probably might have to be modified

    main_ds.attrs['featureType'] = 'trajectory'
    if stationOrSiteName != None:
        main_ds.attrs['title'] = 'Measurements from {} with ship identifier number {}'.format(stationOrSiteName, shipOrMobileLandStationIdentifier) #
        main_ds.attrs['id'] = str(stationOrSiteName) + ', ' + str(shipOrMobileLandStationIdentifier)
    else:
        main_ds.attrs['title'] = 'Measurement from moving station with identifier number {}'.format(shipOrMobileLandStationIdentifier)
        main_ds.attrs['id'] = str(shipOrMobileLandStationIdentifier)
    main_ds.attrs['naming_authority'] = 'World Meteorological Organization'
    main_ds.attrs['source'] = cfg['output']['source']
    main_ds.attrs['summary'] = cfg['output']['abstract']
    today = date.today()

    main_ds.attrs['date_created'] = today.strftime('%Y-%m-%d %H:%M:%S')
    main_ds.attrs['geospatial_lat_min'] = '{:.3f}'.format(main_ds['latitude'].values.min())
    main_ds.attrs['geospatial_lat_max'] = '{:.3f}'.format(main_ds['latitude'].values.max())
    main_ds.attrs['geospatial_lon_min'] = '{:.3f}'.format(main_ds['longitude'].values.min())
    main_ds.attrs['geospatial_lon_max'] = '{:.3f}'.format(main_ds['longitude'].values.max())
    main_ds.attrs['time_coverage_start'] = main_ds['time'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S') # note that the datetime is changed to microsecond precision from nanosecon precision
    main_ds.attrs['time_coverage_end'] = main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')
    
    duration_years = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).years)
    duration_months = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).months)
    duration_days = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).days)
    duration_hours = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).hours)
    duration_minutes = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).minutes)
    duration_seconds = str(relativedelta(main_ds['time'].values[-1].astype('datetime64[s]').astype(datetime), main_ds['time'].values[0].astype('datetime64[s]').astype(datetime)).seconds)
    main_ds.attrs['time_coverage_duration'] = ('P' + duration_years + 'Y' + duration_months +
                                               'M' + duration_days + 'DT' + duration_hours + 
                                               'H' + duration_minutes + 'M' + duration_seconds + 'S')    
    
    main_ds.attrs['keywords'] = keywords
    main_ds.attrs['keywords_vocabulary'] = keywords_voc
    main_ds.attrs['standard_name_vocabulary'] = 'CF Standard Name V79'
    main_ds.attrs['Conventions'] = 'ACDD-1.3, CF-1.6'
    main_ds.attrs['creator_type'] = cfg['author']['creator_type']
    main_ds.attrs['institution'] = cfg['author']['PrincipalInvestigatorOrganisation']
    main_ds.attrs['creator_name'] = cfg['author']['PrincipalInvestigator']
    main_ds.attrs['creator_email'] = cfg['author']['PrincipalInvestigatorEmail']
    main_ds.attrs['creator_url'] = cfg['author']['PrincipalInvestigatorOrganisationURL']
    main_ds.attrs['publisher_name'] = cfg['author']['Publisher']
    main_ds.attrs['publisher_email'] = cfg['author']['PublisherEmail']
    main_ds.attrs['publisher_url'] = cfg['author']['PublisherURL']
    main_ds.attrs['project'] = cfg['author']['Project']
    main_ds.attrs['license'] = cfg['author']['License']

    # this didnt work when I had it further up in the code :-) why?????
    main_ds = main_ds.assign({'trajectory': (('name_strlen'),np.array([shipOrMobileLandStationIdentifier]))})

    return main_ds

def set_encoding(ds, fill=-9999, time_name = 'time', time_units='seconds since 1970-01-01 00:00:00'):
    
    all_encode = {}
        
    for v in list(ds.keys()):
        #if v == 'softwareVersionNumber':
         #   encode = {'zlib': True, 'complevel': 9, '_FillValue':fill}

        if 'float' in str(ds[v].dtype):
            dtip = 'f4'
            encode = {'zlib': True, 'complevel': 9, 'dtype': dtip, '_FillValue':fill}
        elif 'int' in str(ds[v].dtype):
            dtip = 'i4'
            encode = {'zlib': True, 'complevel': 9, 'dtype': dtip, '_FillValue':fill}
        else:
            encode = {'zlib': True, 'complevel': 9}
            
        all_encode[v] = encode
    all_encode['latitude'] = {'zlib': True, 'complevel':9, 'dtype': 'f4', '_FillValue': fill}
    all_encode['longitude'] = {'zlib': True, 'complevel':9, 'dtype': 'f4', '_FillValue': fill}
    all_encode['time'] = {'zlib': True, 'complevel':9, 'dtype': 'i4', '_FillValue': fill}
    return all_encode

def saving_grace(file, key, destdir):
    print(file)
    #sys.exit()
    file = file.sortby('time')
    gb = file.groupby('time.month')
    
    for group_name, group_da in gb:
        #print(group_name,group_da)
        #sys.exit()
        #group_da = file
        t1 = group_da.time.isel(time=0).values.astype('datetime64[s]')
        t1 = t1.astype(datetime)

        t2 = group_da.time.isel(time=-1).values.astype('datetime64[s]')
        t2 = t2.astype(datetime)
        
        timestring1 = t1.strftime('%Y%m%dT%H%M%S')
        timestring2 = t2.strftime('%Y%m%dT%H%M%S')


        ds_dictio = group_da.to_dict()
        bad_time = ds_dictio['coords']['time']['data']
        ds_dictio['coords']['time']['data'] = np.array([((ti - datetime(1970,1,1)).total_seconds()) for ti in bad_time]).astype('i4')
        ds_dictio['coords']['longitude']['data'] = np.float32(ds_dictio['coords']['longitude']['data'])
        ds_dictio['coords']['latitude']['data'] = np.float32(ds_dictio['coords']['latitude']['data'])
        all_ds_station_period = xr.Dataset.from_dict(ds_dictio)
        
        all_ds_station_period = all_ds_station_period.fillna(-9999)
        all_ds_station_period['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
        all_ds_station_period['time'].attrs['standard_name'] = 'time'
        all_ds_station_period['time'].attrs['long_name'] = 'time'
        all_ds_station_period['time'].attrs['coverage_content_type'] = 'referenceInformation'
        
        all_ds_station_period.attrs['id'] = 'syno_{}_{}-{}.nc'.format(key, timestring1, timestring2)
        all_ds_station_period.attrs['title'] = 'Measurements from station with station identifier number {}'.format(key)
        if 'history' not in all_ds_station_period.attrs.keys():
            all_ds_station_period.attrs['history'] = '{} converted syno_{}_{}-{} from BUFR to NetCDF-CF'.format(time.strftime('%Y-%m-%d %H:%M:%S'), key, timestring1, timestring2)
        else:
            all_ds_station_period.attrs['history'] = all_ds_station_period.attrs['history'] + ', \n' + '{} converted syno_{}_{}-{} from BUFR to NetCDF-CF'.format(time.strftime('%Y-%m-%d %H:%M:%S'), key, timestring1, timestring2)
        all_ds_station_period.to_netcdf('{}/syno_{}_{}-{}.nc'.format(destdir, key, timestring1, timestring2),
                                            engine='netcdf4', encoding=set_encoding(all_ds_station_period))
def rename_duplicated_columns(mydf):
    """
    The purpose is to rename duplicated column names in a data frame with .1, .2 etc.
    """

    # Create a list of new, unique column names
    new_column_names = []
    counts = {}
    
    for col in mydf.columns:
        if col in counts:
            # If duplicated, append a counter
            counts[col] += 1
            new_column_names.append(f"{col}_{counts[col]}")
        else:
            # If first occurrence, add name to counts and keep original name
            counts[col] = 0
            new_column_names.append(col)

    # Assign the new list to df.columns
    mydf.columns = new_column_names

    return mydf
        
if __name__ == "__main__":
    """
    Main method below, used when running the software.
    """
    parse = parse_arguments()
    cfg = parse_cfg(parse.cfgfile)
    destdir = cfg['output']['destdir']
    frompath = cfg['station_info']['path']
    stationtype = parse.stationtype
    if parse.initialize:
        """
        If downloading all data do something
        """
        sorted_files = sorting_hat(get_files_initialize(frompath))
        #print(sorted_files)
        if parse.stationtype != 'ship':
            for key, val in sorted_files.items():
                file = block_wigos_state(sorted_files['{}'.format(key)])
                #print(file)
                #sys.exit()
                saving = saving_grace(file, key, destdir)
        else:
            for key, val in sorted_files.items():
                file = shipOrMobileLandStationIdentifier(sorted_files['{}'.format(key)])
                saving = saving_grace(file, key, destdir)
        
               
    elif parse.update:
        """
        Just updating data extraction
        """
        print('updating...')
        
        # find the latest date that is stored in the folder
        
        stations = {}
        stations_names = []
        # get list of stationname in folder
        for file in os.listdir(destdir):
            if file.endswith('.nc'):
                file_split = file.split('_')
                if parse.stationtype == 'block':
                    if len(file_split[1]) == 5:
                        if file_split[1] not in stations:
                            stations[file_split[1]] = [[file_split[2].split('-')[0], file_split[2].split('-')[1][:-3]]]
                            stations_names.append(file_split[1])
                        else:
                            stations[file_split[1]].append([file_split[2].split('-')[0], file_split[2].split('-')[1][:-3]])
                    else:
                        continue
                        
                elif parse.stationtype == 'wigos':
                    print(file_split[1].split('-'))
                    if len(file_split[1].split('-')) > 3 and '-' in file_split[1]:
                        if file_split[1] not in stations:
                            stations[file_split[1]] = [[file_split[2].split('-')[0], file_split[2].split('-')[1][:-3]]]
                            stations_names.append(file_split[1])
                        else:
                            stations[file_split[1]].append([file_split[2].split('-')[0], file_split[2].split('-')[1][:-3]])
                        
                    else:
                        continue
                        
                elif parse.stationtype == 'state':
                    if len(file_split[1].split('-'))==2 and '-' in file_split[1]:
                        if file_split[1] not in stations:
                            stations[file_split[1]] = [[file_split[2].split('-')[0], file_split[2].split('-')[1][:-3]]]
                            stations_names.append(file_split[1])
                        else:
                            stations[file_split[1]].append([file_split[2].split('-')[0], file_split[2].split('-')[1][:-3]])
                    else:
                        continue
                elif parse.stationtype == 'ship':
                    if len(file_split[1]) > 5 and '-' not in file_split[1]:
                        if file_split[1] not in stations:
                            stations[file_split[1]] = [[file_split[2].split('-')[0], file_split[2].split('-')[1][:-3]]]
                            stations_names.append(file_split[1])
                        else:
                            stations[file_split[1]].append([file_split[2].split('-')[0], file_split[2].split('-')[1][:-3]])
                    else:
                        continue
                else:
                    print('must specify stationtype')
                    sys.exit()
        if parse.all_stations == True:            
            all_files = (return_list_of_stations(get_files_specified_dates(parse_cfg(parse.cfgfile)['station_info']['path'])))
        elif parse.spec_station:
            all_files = parse.spec_station
            

        for i in all_files:
            if i in stations_names:
                sorted_files = sorting_hat(get_files_specified_dates(parse_cfg(parse.cfgfile)['station_info']['path']), stations = [str(i)])
                if parse.stationtype != 'ship':
                    file = block_wigos_state(sorted_files['{}'.format(i)])
                else:
                    file = shipOrMobileLandStationIdentifier(sorted_files['{}'.format(key)])
                
                # figuring out which is the last file for that station
                
                end_dates = []
                for k in stations[i]:
                    end_dates.append(datetime.strptime(k[1],'%Y%m%dT%H%M%S'))
                
                last_date = max(end_dates).strftime('%Y%m%dT%H%M%S')
                
                for k in stations[i]:
                    if last_date in k:
                        startday, endday = k
                
                #open latest dataset
                last_file = xr.open_dataset("{}/syno_{}_{}-{}.nc".format(destdir, i, startday, endday),decode_times=False)
                
                # merging
                new_file = xr.merge([file, last_file],compat ='override')
                print(new_file)
                sys.exit()
                #finally remove the old file from destdir and save the new one
                last_file.close()
                os.remove("{}/syno_{}_{}-{}.nc".format(destdir, i, startday, endday))
                
                #lastly save
                saving = saving_grace(new_file, i, destdir)
                
            else:
                continue

                
    elif parse.startday != None:
        print('creating files from {}'.format(parse.startday))
        
        sorted_files = sorting_hat(get_files_specified_dates(frompath))
        print(sorted_files.keys)
        for key,val in sorted_files.items():
            if parse.stationtype != 'ship':
                file = block_wigos_state(sorted_files['{}'.format(key)])
            else:
                file = shipOrMobileLandStationIdentifier(sorted_files['{}'.format(key)])
            saving = saving_grace(file, key, destdir)
            sys.exit()
        
    else:
        print('either type in -i, -u or enter starday/endday')
        sys.exit()
        
