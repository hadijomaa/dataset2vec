#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:14:11 2020

@author: hsjomaa
"""

from sklearn.metrics import pairwise as pairw
import numpy as np
import argparse
import json
import pandas as pd
import os
import sys
from helper_fn import regret
np.random.seed(327)

parser = argparse.ArgumentParser()
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--method', help='Metafeatures used', choices=['aaai','tstr','d2v'],default='aaai')
parser.add_argument('--searchspace', type=str,default='a')
args = parser.parse_args()

currentdir     = os.path.dirname(os.path.realpath(__file__))
rootdir   = '/'.join(currentdir.split('/')[:-1])
sys.path.insert(0,rootdir)
from dataset import Dataset

savedir     = os.path.join(rootdir,"results","nearestneighbor",args.method,f"searchspace-{args.searchspace}",f"split-{args.split}")
os.makedirs(savedir,exist_ok=True)


splits_file       = os.path.join(rootdir, "metadataset",f"searchspace-{args.searchspace}",f"searchspace-{args.searchspace}-splits.csv")
metasplits        = pd.read_csv(splits_file,index_col=0)

info_file       = os.path.join(rootdir, "metadataset"  ,"info.json")
configuration   = json.load(open(info_file,'r'))[args.searchspace]

# specific to rgpe
configuration['split']                     = args.split
configuration['searchspace']               = args.searchspace
configuration['minmax']                    = True

normalized_dataset         = Dataset(configuration,rootdir,use_valid=True)
metafeatures = {'aaai':pd.read_csv(os.path.join(rootdir,"metafeatures","mf1.csv"),index_col=0,header=0),
                'tstr':pd.read_csv(os.path.join(rootdir,"metafeatures","mf2.csv"),index_col=0,header=0),
                'd2v':pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{args.split}.csv"),index_col=0,header=0)}

z          = metafeatures[args.method]
source     = np.asarray(z)
files = metasplits[f"train-{args.split}"].dropna().tolist()+metasplits[f"valid-{args.split}"].dropna().tolist()+metasplits[f"test-{args.split}"].dropna().tolist()

for file in files:
    response = normalized_dataset.global_surr[file][:,0]

    targetmf   = np.asarray(z.loc[file])
    
    distance = pairw.euclidean_distances(targetmf[None],source).ravel()
    
    x  = np.argsort(distance).reshape(-1,)
    assert(z.index[x[0]]==file)
    x  = x[1:] # get rid of similar dataset
    
    yr = normalized_dataset.global_surr[z.index[x[0]]][:,0]
    
    x  = np.argsort(yr)[::-1]
    
    y = np.vstack([response[_] for _ in x]).reshape(-1,)
    results            = regret(y,response)
    results.to_csv(os.path.join(savedir,f"{file}.csv"))    