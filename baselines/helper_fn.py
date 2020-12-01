#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 19:30:54 2020

@author: hsjomaa
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise as pairw

def regret(output,response):
    incumbent   = output[0]
    best_output = []
    for _ in output:
        incumbent = _ if _ > incumbent else incumbent
        best_output.append(incumbent)
    opt       = max(response)
    orde      = list(np.sort(np.unique(response))[::-1])
    tmp       = pd.DataFrame(best_output,columns=['regret_validation'])
    
    tmp['rank_valid']        = tmp['regret_validation'].map(lambda x : orde.index(x))
    tmp['regret_validation'] = opt - tmp['regret_validation']
    return tmp

def warm_start(method,file,dataset,k,metafeatures,zusatz_ordner=None):
    '''
    

    Parameters
    ----------
    method : str
        metafeatures used for warm-starting.
    file : str
        file name.
    dataset : Dataset
        dataset.
    k : int
        number of initial points.
    fold : str
        which fold is the file from (train,test,valid).
    position : int
        position of file in respective split.
    zusatz_ordner : str
        directory for transferable.

    Returns
    -------
    x : list
        indices of selected hyper-parameters.

    '''
    
    if method in ['d2v','tstr','aaai']:

        z          = metafeatures[method]
        
        source     = np.asarray(z) 
        
        targetmf   = np.asarray(z.loc[file])
        
        distance = pairw.euclidean_distances(targetmf[None],source).ravel()
        
        x  = np.argsort(distance).reshape(-1,)
        
        assert(z.index[x[0]]==file)
        
        x  = x[1:] # get rid of similar dataset
        
        yr = dataset.global_surr[z.index[x[0]]][:,0]
        
        x  = np.argsort(yr)[::-1] # large to small
        
        x  = x[:k]
        
    elif method in ["random"]:
        x = np.random.choice(np.arange(0,dataset.cardinality),size=k,replace=False)
    elif method in ["transferable"]:
        x = np.asarray(pd.read_csv(zusatz_ordner,header=0,index_col=0)).reshape(-1,)[:k]
    return x
