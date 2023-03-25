#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:28:07 2022

@author: nick
"""
import sys
import pickle5 as pickle
import pandas as pd
import json
import requests
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
from pandas.io.sql import get_schema
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

'''
Get metadata about the run for building dataframe, this is where 
the user has to be able to connect o the original run folder to access the images, kml, and txt files
'''
def getRunMetadata(rootdir):
    
    _date,runNum = rootdir.split('/')[-1].split('_')[:2]
    runNum = int(runNum)
    date = '{}/{}/{}'.format(_date[4:6],_date[-2:],_date[:4])
    list_subfolders_with_paths = [f.path for f in os.scandir(rootdir) if f.is_dir()]
    
    return runNum,date,list_subfolders_with_paths


'''
Process the pickle files for each job and write to database
'''
def processJobPickles(rootdir,dbPath=None):
    
    # Get metadata for the run
    runNum,date,_ = getRunMetadata(rootdir)
    pklFilenames = filter(lambda x: x[-4:]=='.pkl', os.listdir(rootdir))
    
    if not dbPath:
        dbPath = "sqlite+pysqlite:///detections.db"
        
    engine = create_engine(dbPath)
    
    #os.mkdir('run{}_pairs'.format(runNum))

    for pklFilename in pklFilenames:
        
        print('Loading {}...'.format(pklFilename))

        # Load pickle for the job
        with open(os.path.join(rootdir,pklFilename),'rb') as f:
            pkldata = pickle.load(f)
        # pkldata = pd.read_pickle(os.path.join(rootdir,pklFilename))
        
        # Extract the subfolder name which contains the images and lat/lon data
        # If it fails then there was no detections
        try:
            subfolder = pkldata[0][0].split('/')[-2].split('_')[0]
        except:
            continue

        # Convert pickle contents to dataframe
        df = jobPickleToDataframe(pkldata,(runNum,date,subfolder))
        
        # Write to database table
        # result = toDB(df,engine)

        # Filter high confidence detections and compute pairs
        hcDf = df[df.Conf>=.8]
        createOrAppendDataframe(hcDf,'DetectionDfs/Run{}_Detections.pkl'.format(runNum))
        
        # pairs = runDfToPairs(hcDf)
        # pairs.to_pickle(os.path.join('run{}_pairs'.format(runNum),'{}_pairs.pkl').format(subfolder))
        # pairsResult = pairsToDB(pairs,engine)

'''
Check if a dataframe already exists at the path. If so, append the new data. Otherwise, make the file.
'''
def createOrAppendDataframe(df,path):

    if os.path.exists(path):

        _df = pd.read_pickle(path)
        _df = pd.concat([_df,df])
        _df.to_pickle(path)
    
    else:
        df.to_pickle(path)

'''
Write dataframe to database table
'''
def toDB(df,engine):
    
    with engine.connect() as conn:
        result = df.to_sql('Pedestrians',conn,if_exists='append',index=False)

    return result

'''
Write pairs dataframe to database table
'''
def pairsToDB(df,engine):

    with engine.connect() as conn:
        result = df.to_sql('Pairs',conn,if_exists='append',index=False)

    return result

'''
Convert a single job-batch pickle file into a dataframe
'''
def jobPickleToDataframe(pkldata,metadata):
    
    # print(f'# Detections: {len(pkldata)}')

    # Unpack metadata
    runNum,date,subfolder = metadata
     
    # Load lat/lon data for the subfolder
    base = #put where your base folder for all your runs are under

    runFolders = os.listdir(base)
    runFolder = #write a function similar to the filter commented below that contains the structure of the folder AFTER the YYYYMMDD_XX naming structure
    #next(filter(lambda x:'seattle_fullcity_processed' in x and int(x.split('_')[1])==int(runNum), runFolders))
    runPath = os.path.join(base,runFolder)    

   frameFilename = os.path.join(base,runFolder,subfolder,_subfolder + "_framepos") 
    try:
        txtdata = pd.read_csv(frameFilename+".txt")
    except FileNotFoundError:
        txtdata = pd.read_csv(frameFilename+".kml")

    # Construct rows of new dataframe
    subfolder = subfolder.split('/')[-1]
    rowList = []
    for row in pkldata:
        rowList.append([runNum,'NA',subfolder,row[0][-10:-6],row[0][-5],date,txtdata.iat[int(row[0][-10:-6]),2],
                     txtdata.iat[int(row[0][-10:-6]),3],txtdata.iat[int(row[0][-10:-6]),4],row[1],row[2],row[3],row[4],row[6]])
    
    # Construct dataframe
    df = pd.DataFrame(rowList,columns=['Run','MA','Subfolder','Filename','Side','Date','Lat','Lon','Altitude','Xmin','Ymin','Xmax','Ymax','Conf'])
    # Perform GeoID Matching
    df = geoIdMatching(df)

    return df



if __name__ == "__main__":
    
    args = sys.argv
    
    if len(args)==2:
        _,rootdir = args
        args = (rootdir,)
    elif len(args)==3:
        _,rootdir,dbPath = args
        args = (rootdir,dbPath)
    else:
        raise Exception('Too many arguments! Expected 1 or 2, but got {}'.format(len(args)))
        
    processJobPickles(*args)
        
