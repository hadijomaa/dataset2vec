#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:56:16 2020

@author: hsjomaa
"""

import numpy as np
import argparse
import pandas as pd
import os
import sys
import json
np.random.seed(327)

parser = argparse.ArgumentParser()
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--method', help='Metafeatures used', choices=['aaai','tstr','d2v'],default='d2v')
parser.add_argument('--searchspace', type=str,default='a')

args = parser.parse_args()

currentdir     = os.path.dirname(os.path.realpath(__file__))
rootdir   = '/'.join(currentdir.split('/')[:-1])
sys.path.insert(0,rootdir)
from dataset import Dataset

savedir     = os.path.join(rootdir,"baselines","datasets","two-stage-surrogate",f"searchspace-{args.searchspace}",args.method,f"split-{args.split}" if args.method=="d2v" else "")
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
    
files = metasplits[f"train-{args.split}"].dropna().tolist()+metasplits[f"valid-{args.split}"].dropna().tolist()+metasplits[f"test-{args.split}"].dropna().tolist()

for file in files:
    z            = np.asarray(metafeatures[args.method].loc[file]).reshape(1,-1)
    z            =  pd.DataFrame(np.repeat(z,normalized_dataset.cardinality,axis=0))
    data    = pd.concat([pd.DataFrame(normalized_dataset.global_surr[file]),z],axis=1)

    data.to_csv(os.path.join(savedir,file),header=False,index=False,sep=' ')
    break