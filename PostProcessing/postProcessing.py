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
Get metadata about the run for building dataframe
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
    pulsar_base = '/home/jupyter/projects/PRJ-2759/Processed_Data/Pulsar/'

    runFolders = os.listdir(pulsar_base)
    runFolder = next(filter(lambda x:'seattle_fullcity_processed' in x and int(x.split('_')[1])==int(runNum), runFolders))
    runPath = os.path.join(pulsar_base,runFolder)    

    # If there are afternoon/morning subfolders, list those subdirs
    hasSubfolders = False
    for item in os.listdir(runPath):
        if 'Afternoon' in item:
            hasSubfolders = True

    if hasSubfolders:
        subfolders = []
        for fold in os.listdir(runPath):
             if os.path.isdir(os.path.join(runPath,fold)):
                 for fi in os.listdir(os.path.join(runPath,fold)):
                    subfolders.append(os.path.join(fold,fi))

        subfolder = next(filter(lambda x:subfolder in x and '.txt' not in x,subfolders)) 
        
        _subfolder = subfolder[:-7] if '_Output' in subfolder else subfolder
            
        if 'legacy' in subfolder.split('-'):
            tst = _subfolder.split('-')[0] + '-' + _subfolder.split('-')[1]
            frameFilename = os.path.join(pulsar_base,runFolder,subfolder,tst + "_framepos")
        else: 
            frameFilename = os.path.join(pulsar_base,runFolder,subfolder,_subfolder.split('/')[-1] + "_framepos")
            #frameFilename = os.path.join(pulsar_base,runFolder,subfolder,subfolder.split('/')[-1] + "_framepos")

    else:
        
        subfolder = next(filter(lambda x:subfolder in x and '.txt' not in x,os.listdir(runPath))) 
        
        _subfolder = subfolder[:-7] if '_Output' in subfolder else subfolder
        
        if 'legacy' in subfolder.split('-'):
            tst = _subfolder.split('-')[0] + '-' + _subfolder.split('-')[1]
            frameFilename = os.path.join(pulsar_base,runFolder,subfolder,tst + "_framepos")
        else:
            frameFilename = os.path.join(pulsar_base,runFolder,subfolder,_subfolder + "_framepos")
            #frameFilename = os.path.join(pulsar_base,runFolder,subfolder,subfolder + "_framepos")
    
    try:
        txtdata = pd.read_csv(frameFilename+".txt")
    except FileNotFoundError:
        txtdata = pd.read_csv(frameFilename+".kml")

    # Construct rows of new dataframe
    subfolder = subfolder.split('/')[-1] if hasSubfolders else subfolder
    rowList = []
    for row in pkldata:
        rowList.append([runNum,'NA',subfolder,row[0][-10:-6],row[0][-5],date,txtdata.iat[int(row[0][-10:-6]),2],
                     txtdata.iat[int(row[0][-10:-6]),3],txtdata.iat[int(row[0][-10:-6]),4],row[1],row[2],row[3],row[4],row[6]])
    
    # Construct dataframe
    df = pd.DataFrame(rowList,columns=['Run','MA','Subfolder','Filename','Side','Date','Lat','Lon','Altitude','Xmin','Ymin','Xmax','Ymax','Conf'])
    # Perform GeoID Matching
    df = geoIdMatching(df)

    return df

'''
Cross reference lat/lon with .shp files
'''
def geoIdMatching(df):
    
    census_tracts = gpd.read_file('/home/jupyter/projects/PRJ-2759/Computer_Vision/Final_Labels/kc_bg_10.shp')
    
    points_df = df
    
    geometry = [Point(xy) for xy in zip(points_df.Lon, points_df.Lat)]
    crs = 'epsg:4326'
    gdf = gpd.GeoDataFrame(points_df, crs=crs, geometry=geometry)
    gdf = gdf.to_crs('epsg:2926')
    merged_file = gpd.sjoin(gdf, census_tracts, how='left', op='within')
    merged_df = pd.DataFrame(merged_file.drop(['geometry','index_right','INTPTLAT10','INTPTLON10','Shape_Leng','Shape_Area','STATEFP10','COUNTYFP10','TRACTCE10','MTFCC10','ALAND10','AWATER10','FUNCSTAT10','OBJECTID','BLKGRPCE10'],axis=1))
    
    
    IncomeData =  pd.read_csv("IncomeData.csv")
    merged_df = merged_df.drop('TRACT',axis=1)
    merged_df.columns = ['Run', 'MA', 'Subfolder', 'Filename', 'Side', 'Date', 'Lat', 'Lon',
       'Altitude', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'Conf', 'id', 'NAMELSAD10',
        'TRBG']
    merged_df['id'] = merged_df['id'].map(lambda x: x[:-1]).astype(float)
    data3 = pd.merge(left=merged_df,right=IncomeData,on='id',how='outer')
    data4 = pd.merge(left=merged_df,right=IncomeData,on='id',how='inner')
    data4 = data4.drop('Unnamed: 0',axis=1)
 
    data4.columns = ['Run', 'MA', 'Subfolder', 'Filename', 'Side', 'Date', 'Lat', 'Lon',
       'Altitude', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'Conf', 'TractID', 'NAMELSAD10',
        'TRBG', 'MedianIncome', 'IncomeBracket']

    return data4

'''
Take a results dataframe for a single survey and compute each pair of pedestrians
for pairwise distance estimation.
'''
def runDfToPairs(df):
    
    df = df.reset_index()
    mergeCol = ['Run','Subfolder','Filename','Side']
    
    # Assign an index to each image
    df['img_id'] = df.groupby(mergeCol).ngroup()
    
    # Calculate each pair of pedestrians using cross join
    comb = pd.merge(df[['index','img_id']],df[['index','img_id']],
                    on='img_id',suffixes=('1','2'))
    comb = comb[comb.index1 < comb.index2]
    
    # Merge in the file info and bounding boxes to streamline loading images for distance calculations
    mc = ['index','Run','Subfolder','Filename','Side','Xmin', 'Ymin', 'Xmax', 'Ymax']
    comb = pd.merge(comb,df[mc],left_on='index1',right_on='index')
    comb = pd.merge(comb,df[['index','Xmin', 'Ymin', 'Xmax', 'Ymax']],
                    left_on='index2',right_on='index',suffixes=('1','2'))
    comb = comb.loc[:,~comb.columns.duplicated()].copy()
    
    # Reorder columns for neatness
    comb = comb[['img_id', 'Run', 'Subfolder', 'Filename', 'Side',
       'index1','Xmin1', 'Ymin1', 'Xmax1', 'Ymax1', 'index2', 'Xmin2', 'Ymin2', 'Xmax2', 'Ymax2']]
    
    # Add a distance column
    comb['distance'] = np.nan
    
    return comb

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
        
