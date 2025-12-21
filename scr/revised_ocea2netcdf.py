#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from eccodes import *
import xarray as xr
import argparse
import yaml
from datetime import datetime, timedelta, date
import json
import subprocess
import numpy as np 
import pandas as pd
from funcs.useful_functions import cf_match
#from funcs.code_tables import table_002032, table_002033
from funcs.get_keywords import *
import re
from dateutil.relativedelta import relativedelta


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--cfg", dest='cfgfile',
                        help="Configuration file", required=True)
    parser.add_argument("-col", "--column", dest='column_name', choices=['ship', 'buoy'],
                        help="Measurement type name to use for grouping DataFrames ('ship' or 'buoy')", required=True)
    args = parser.parse_args()

    if args.cfgfile is False:
        parser.print_help()
        parser.exit()

    return args

def parse_cfg(cfgfile):
    # Read config file
    print('Reading configuration file:', cfgfile)
    with open(cfgfile, 'r') as ymlfile:
        cfgstr = yaml.full_load(ymlfile)

    return cfgstr

def initialize_files(cfgstr):
    files = []
    data = []
    print('Initializing files from input path:', cfgstr['input']['path'])
    for file in os.listdir(cfgstr['input']['path']):
        if file.endswith('.bufr'):
            file_path = ('{}/{}'.format(cfgstr['input']['path'], file))
            files.append(datetime.strptime(file[5:-5], "%Y%m%d%H"))
            data.append(file_path)
            #print('Found BUFR file:', file_path)

    return files, data
     

def read_bufr_messages(file):
    messages = []
    with open(file, 'rb') as f:
        nmsg = codes_count_in_file(f)
        for i in range(nmsg):
            bufr = codes_bufr_new_from_file(f)
            if bufr is None:
                print(f'Failed to load BUFR message from {file}')
                continue
            try:
                codes_set(bufr, 'unpack', 1)  # Unpack the BUFR message
            except CodesInternalError as err:
                print(f"Error decoding BUFR message from {file}: {err}")
                codes_release(bufr)
                continue
            messages.append(bufr)
    return messages


def get_all_keys(bufr):
    subset = 0
    keys = []
    try:
        iter_id = codes_bufr_keys_iterator_new(bufr)  
    except CodesInternalError as err:
        print(f"Error creating keys iterator: {err}")
        return keys

    while codes_bufr_keys_iterator_next(iter_id):
        key_name = codes_bufr_keys_iterator_get_name(iter_id)
        keys.append(key_name)
        if key_name == "subsetNumber":
            subset += 1
            

    codes_bufr_keys_iterator_delete(iter_id)
    
    return keys


def decode_bufr_message(bufr):
    keys = get_all_keys(bufr)
    data = {}
    for key in keys:
        try:
            value = codes_get(bufr, key)
            data[key] = value           
            
        except CodesInternalError as err:
            print('Error with key="%s" : %s' % (key, err.msg))
            if 'Passed array is too small' in str(err):
                try:
                    value = codes_get_array(bufr, key)
                    data[key] = value

                except Exception as array_exception:
                    print(f'Warning: Could not extract array key {key} (array_exception): {str(array_exception)}')


    return data


def merge_columns(decoded_dict):
    """
    Merge columns with matching names into a single column.
    """
    merged_data = {}
    for key in decoded_dict:
        # Removing the prefix (e.g., "#1#", "#2#", etc.)
        base_name = key.split('#')[-1]
        if base_name not in merged_data:
            merged_data[base_name] = []
        value = decoded_dict[key]
        if isinstance(value, list):
            merged_data[base_name].extend(value)
        else:
            merged_data[base_name].append(value)
    
    
    return merged_data     




def dict_to_dataframe(decoded_dict):
    """
    Convert a decoded dictionary to a pandas.DataFrame.
    """
    # Merge columns with matching names
    merged_data_dict = merge_columns(decoded_dict)

    # Determine the maximum length of the values lists
    max_len = max(len(value) if isinstance(value, list) else 1 for value in merged_data_dict.values())
    
    # Convert merged_data_dict to DataFrame
    data_dict = {key: (value + [None] * (max_len - len(value))) if isinstance(value, list) else [value] + [None] * (max_len - 1) for key, value in merged_data_dict.items()}
    
    # Create a DataFrame from the data_dict
    df = pd.DataFrame(data_dict)

    return df


def find_matching_dataframes(dataframes, column_name):
    """
    Find and append DataFrames that have the same value in a given column.
    """
    grouped_dataframes = {}

    for dataframe_dict in dataframes:
        for file_path, df in dataframe_dict.items():
            if column_name in df.columns:
                for value in df[column_name].unique():
                    if value not in grouped_dataframes:
                        grouped_dataframes[value] = pd.DataFrame()
                    grouped_dataframes[value] = pd.concat(
                        [grouped_dataframes[value], df[df[column_name] == value]],
                        ignore_index=True
                    )
            else:
                print(f"Column '{column_name}' not found in DataFrame from file: {file_path}")

    return grouped_dataframes


def convert_to_xarray(dataset_dict):
    """
    Convert each concatenated DataFrame to an xarray.Dataset and assign coordinates.
    """
    dict_datasets = {}
    
    for key, df in dataset_dict.items():
        # Select the last 15 columns of the DataFrame
        df_filtered = df.iloc[:, -15:]

        # Convert DataFrame to xarray.Dataset
        ds = xr.Dataset.from_dataframe(df_filtered)
        
        # Create 'time' coordinate from year, month, day, hour, and minute
        if all(col in ds for col in ['year', 'month', 'day', 'hour', 'minute']):
            times = pd.to_datetime({
                'year': ds['year'].values,
                'month': ds['month'].values,
                'day': ds['day'].values,
                'hour': ds['hour'].values,
                'minute': ds['minute'].values
            })
            ds = ds.assign_coords(time=('index', times))
            ds = ds.swap_dims({'index': 'time'})
            ds = ds.sortby('time')
        
        # Assign latitude and longitude as coordinates
        if 'latitude' in ds and 'longitude' in ds:
            ds = ds.assign_coords(latitude=('index', ds['latitude'].values))
            ds = ds.assign_coords(longitude=('index', ds['longitude'].values))
        
        # Ensure all variables have consistent and compatible data types
        for var in ds.data_vars:
            if ds[var].dtype == object:
                try:
                    ds[var] = ds[var].astype(str)  # Convert to string if mixed types
                except Exception as e:
                    print(f"Could not convert variable {var} to string: {e}")
                    ds = ds.drop_vars(var)  # Drop the variable if conversion fails
        
        dict_datasets[key] = ds
    
    return dict_datasets

def set_attrs_ship(dataset_dict):
    """
    Add specific shipStation variables.
    """

    for key, ds in dataset_dict.items():
        for var in ds.keys():
            # adding coverage_content_type
            thematicClassification = ['shipOrMobileLandStationIdentifier', 'stationType']
            if var in thematicClassification:
                ds[var].attrs['coverage_content_type'] = 'thematicClassification'
            else:
                ds[var].attrs['coverage_content_type'] = 'physicalMeasurement'

            
            if 'time' in ds.coords:
                ds.attrs['time_coverage_start'] = ds.coords['time'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S') 
                ds.attrs['time_coverage_end'] = ds.coords['time'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate the time_coverage_duration
                duration = ds.coords['time'].values[-1].astype('datetime64[s]') - ds.coords['time'].values[0].astype('datetime64[s]')
                duration_seconds = duration / np.timedelta64(1, 's')
                days, seconds = divmod(duration_seconds, 86400)
                hours, seconds = divmod(seconds, 3600)
                minutes, seconds = divmod(seconds, 60)

                # Format the duration as an ISO 8601 duration string
                duration_str = f"P{int(days)}DT{int(hours)}H{int(minutes)}M{int(seconds)}S"
                ds.attrs['time_coverage_duration'] = duration_str
            

def set_attrs_buoy(dataset_dict):
    """
    Add specific buoyPlatform variables.
    """

    for key, ds in dataset_dict.items():
        for var in ds.keys():
            if 'time' in ds.coords and ds.coords['time'].size > 0:
                #print(ds.coords['time'])
                ds.attrs['time_coverage_start'] = ds.coords['time'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S') 
                ds.attrs['time_coverage_end'] = ds.coords['time'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate the time_coverage_duration
                duration = ds.coords['time'].values[-1].astype('datetime64[s]') - ds.coords['time'].values[0].astype('datetime64[s]')
                duration_seconds = duration / np.timedelta64(1, 's')
                days, seconds = divmod(duration_seconds, 86400)
                hours, seconds = divmod(seconds, 3600)
                minutes, seconds = divmod(seconds, 60)

                # Format the duration as an ISO 8601 duration string
                duration_str = f"P{int(days)}DT{int(hours)}H{int(minutes)}M{int(seconds)}S"
                ds.attrs['time_coverage_duration'] = duration_str
                



def set_attrs(dataset_dict, cfgstr):
    """
    Add variable and global attributes to the expedition measurements.
    """

    manual_stdname = {'windDirection': 'wind_from_direction', 'dewpointTemperature': 'dew_point_temperature',
                          'heightOfBaseOfCloud':'cloud_base_altitude', 
                          'characteristicOfPressureTendency':'tendency_of_air_pressure',
                          'horizontalVisibility':'visibility_in_air', 
                          'totalPrecipitationOrTotalWaterEquivalent': 'precipitation_amount',
                          'totalSnowDepth':'lwe_thickness_of_snowfall_amount'}

    keywords_list = []

    for key, ds in dataset_dict.items():
        for var in ds.keys():
            change_upper_case = re.sub(r'(?<!^)(?=\d{2})', '_',re.sub(r'(?<!^)(?=[A-Z])', '_', var)).lower() 
            keywords_list.append(change_upper_case)
                        
            """
            #TODO external function cf_match() is not matching any variables to standard_names. Would need to revise function.

            standardname_check = cf_match(change_upper_case)
            if standardname_check != 'fail':
                ds[var].attrs['standard_name'] = standardname_check
            
            if var in manual_stdname.keys():
                ds[var].attrs['standard_name'] = manual_stdname[var]
            """

            # Directly set the standard_name using change_upper_case
            ds[var].attrs['standard_name'] = change_upper_case
            
            # Apply manual standard names if available
            if var in manual_stdname:
                ds[var].attrs['standard_name'] = manual_stdname[var]

            
            # Adding long name
            long_name = re.sub('_',' ', change_upper_case)
            ds[var].attrs['long_name'] = long_name


            units_dict = {
                'shipOrMobileLandStationIdentifier': 'CCITT IA5',
                'latitude': 'deg',
                'longitude': 'deg',
                'indicatorForDigitization': 'CODE TABLE',
                'methodOfSalinityOrDepthMeasurement': 'CODE TABLE',
                'delayedDescriptorReplicationFactor': 'Numeric',
                'depthBelowWaterSurface': 'm',
                'oceanographicWaterTemperature': 'K',
                'salinity': '0/00',
                'windDirection': 'degrees',
                'dewpointTemperature': 'K',
                'heightOfBaseOfCloud': 'm',
                'characteristicOfPressureTendency': 'Pascals per second',
                'horizontalVisibility': 'm',
                'totalPrecipitationOrTotalWaterEquivalent': 'mm',
                'totalSnowDepth': 'm'
            }

            indicator_digi = {
                '0': 'VALUES AT SELECTED DEPTHS (DATA POINTS FIXED BY THE INSTRUMENT OR SELECTED BY ANY OTHER METHOD)',
                '1': 'VALUES AT SIGNIFICANT DEPTHS (DATA POINTS TAKEN FROM TRACES AT SIGNIFICANT DEPTHS)',
                '2': 'RESERVED',
                '3': 'MISSING VALUE'
            }

            method_of_salinity = {
                '0': 'NO SALINITY MEASURED',
                '1': 'IN SITU SENSOR, ACCURACY BETTER THAN 0.02 %',
                '2': 'IN SITU SENSOR, ACCURACY LESS THAN 0.02 %',
                '3': 'SAMPLE ANALYSIS',
                '7': 'MISSING VALUE'
            }

            # Adding units
            if var in units_dict:
                ds[var].attrs['units'] = units_dict[var]

            # Check for indicatorForDigitization and set the unit using the indicator_digi dictionary
            if var == 'indicatorForDigitization':
                indicator_values = ds[var].values.astype(int)
                unique_values = set(indicator_values)
                unit_texts = [indicator_digi.get(str(val), 'UNKNOWN') for val in unique_values]
                ds[var].attrs['units'] = ', '.join(unit_texts)

            # Check for imethodOfSalinityOrDepthMeasurement and set the unit using the method_of_salinity dictionary
            if var == 'methodOfSalinityOrDepthMeasurement':
                indicator_values = ds[var].values.astype(int)
                unique_values = set(indicator_values)
                unit_texts = [method_of_salinity.get(str(val), 'UNKNOWN') for val in unique_values]
                ds[var].attrs['units'] = ', '.join(unit_texts)

            
            ds['longitude'].attrs['long_name'] = 'longitude'
            ds['longitude'].attrs['standard_name'] = 'longitude'
            ds['longitude'].attrs['units'] = 'degrees_east'
            ds['longitude'].attrs['coverage_content_type'] = 'coordinate'

            ds['latitude'].attrs['long_name'] = 'latitude'
            ds['latitude'].attrs['standard_name'] = 'latitude'
            ds['latitude'].attrs['units'] = 'degrees_north'
            ds['latitude'].attrs['coverage_content_type'] = 'coordinate'

            #TODO works manually, so need to evaluate external function code_tables()
            """
            # Check for methodOfSalinityOrDepthMeasurement and set the unit using table_002033 function
            if var == 'methodOfSalinityOrDepthMeasurement':
                indicator_values = ds[var].values
                ds[var].attrs['units'] = table_002033(indicator_values)
            """


            
            ## GLOBAL ATTRS!!
            ds.attrs['featureType'] = 'trajectory'
            ds.attrs['title'] = f'Measurements with identifier {key}.'
            ds.attrs['id'] =str(key)

            #TODO would be nice to implement as distincts between ship/station and buoy/platform
            """
            if shipOrMobileLandStationIdentifier != None:
                main_ds.attrs['title'] = 'Measurements with ship identifier number {}'.format(shipOrMobileLandStationIdentifier) #
                main_ds.attrs['id'] = str(shipOrMobileLandStationIdentifier)
            else:
                main_ds.attrs['title'] = 'Measurement from moving station with identifier number {}'.format(shipOrMobileLandStationIdentifier)
                main_ds.attrs['id'] = str(shipOrMobileLandStationIdentifier)
            """
            ds.attrs['naming_authority'] = 'World Meteorological Organization'
            ds.attrs['source'] = cfgstr['output']['source']
            ds.attrs['summary'] = cfgstr['output']['abstract']
            
            
            today = date.today()
            
            ds.attrs['date_created'] = today.strftime('%Y-%m-%d %H:%M:%S')
            # Ensure latitude attributes
            if 'latitude' in ds and ds['latitude'].size > 0:
                ds['latitude'].attrs['long_name'] = 'latitude'
                ds['latitude'].attrs['standard_name'] = 'latitude'
                ds['latitude'].attrs['units'] = 'degrees_north'
                ds['latitude'].attrs['coverage_content_type'] = 'coordinate'
                ds.attrs['geospatial_lat_min'] = '{:.3f}'.format(ds['latitude'].values.min())
                ds.attrs['geospatial_lat_max'] = '{:.3f}'.format(ds['latitude'].values.max())

            # Ensure longitude attributes
            if 'longitude' in ds and ds['longitude'].size > 0:
                ds['longitude'].attrs['long_name'] = 'longitude'
                ds['longitude'].attrs['standard_name'] = 'longitude'
                ds['longitude'].attrs['units'] = 'degrees_east'
                ds['longitude'].attrs['coverage_content_type'] = 'coordinate'
                ds.attrs['geospatial_lon_min'] = '{:.3f}'.format(ds['longitude'].values.min())
                ds.attrs['geospatial_lon_max'] = '{:.3f}'.format(ds['longitude'].values.max())

            # Adding keywords using external function get_keywords()
            keywords, keywords_voc = get_keywords(keywords_list)
            
            ds.attrs['keywords'] = keywords
            ds.attrs['keywords_vocabulary'] = keywords_voc
            ds.attrs['standard_name_vocabulary'] = 'CF Standard Name V79'
            ds.attrs['Conventions'] = 'ACDD-1.3, CF-1.6'
            ds.attrs['creator_type'] = cfgstr['author']['creator_type']
            ds.attrs['institution'] = cfgstr['author']['PrincipalInvestigatorOrganisation']
            ds.attrs['creator_name'] = cfgstr['author']['PrincipalInvestigator']
            ds.attrs['creator_email'] = cfgstr['author']['PrincipalInvestigatorEmail']
            ds.attrs['creator_url'] = cfgstr['author']['PrincipalInvestigatorOrganisationURL']
            ds.attrs['publisher_name'] = cfgstr['author']['Publisher']
            ds.attrs['publisher_email'] = cfgstr['author']['PublisherEmail']
            ds.attrs['publisher_url'] = cfgstr['author']['PublisherURL']
            ds.attrs['project'] = cfgstr['author']['Project']
            ds.attrs['license'] = cfgstr['author']['License']


            


def main():
    args = parse_arguments()
    cfgstr = parse_cfg(args.cfgfile)
    files, data = initialize_files(cfgstr)
    all_decoded_data = []
    all_dataframes = []

    # Mapping from the aliases to the full measurement type
    COLUMN_NAME_MAPPING = {
    'ship': 'shipOrMobileLandStationIdentifier',
    'buoy': 'buoyOrPlatformIdentifier'
    }

    for file in data:
        bufr_messages = read_bufr_messages(file)
        for bufr in bufr_messages:
            if bufr:
                decoded_data = decode_bufr_message(bufr)
                all_decoded_data.append({file: decoded_data})
                codes_release(bufr)
    

    # Convert each decoded dictionary to a pandas.DataFrame and store in a list
    for item in all_decoded_data:
        for file_path, decoded in item.items():
            #print(f"File: {file_path}")
            dataframe = dict_to_dataframe(decoded)
            all_dataframes.append({file_path: dataframe})
            
    
    # Use the measurement type specified in the command-line arguments
    column_name = COLUMN_NAME_MAPPING[args.column_name]

    # Merging measurements together 
    grouped_dataframes = find_matching_dataframes(all_dataframes, column_name)
    
    # Convert each concatenated DataFrame to a xarray.Dataset
    dict_datasets = convert_to_xarray(grouped_dataframes)

    # Set attributes for each Dataset
    set_attrs(dict_datasets, cfgstr)

    if args.column_name == 'ship':
        set_attrs_ship(dict_datasets)
    elif args.column_name == 'buoy':
        set_attrs_buoy(dict_datasets)


    # Get the output directory from the configuration file
    output_dir = cfgstr['output']['path']

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # Write each Dataset to a NetCDF file
    for value, dataset in dict_datasets.items():
        # Skip datasets where buoyOrPlatformIdentifier is NaN
        if args.column_name == 'buoy' and pd.isna(value):
            print(f"Skipping dataset with {column_name} = NaN")
            continue

        filename = os.path.join(output_dir, f"{value}.nc") 
        print(f"Writing xarray Dataset with {column_name} = {value} to {filename}")
        dataset.to_netcdf(filename)


    for value, dataset in dict_datasets.items():
        print(f"\nXarray Dataset with {column_name} = {value}:")
        print(dataset)
        print("\n")
    

if __name__ == '__main__':
   main()
