#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:57:46 2020

@author: hsjomaa
"""

import numpy as np
import argparse
import json
import pandas as pd
import os
import sys
from helper_fn import regret
np.random.seed(327)
rng = np.random.RandomState(seed=381)
parser = argparse.ArgumentParser()
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--searchspace', type=str,default='c')

args = parser.parse_args()

currentdir     = os.path.dirname(os.path.realpath(__file__))
rootdir   = '/'.join(currentdir.split('/')[:-1])
sys.path.insert(0,rootdir)
from dataset import Dataset

savedir     = os.path.join(rootdir,"results","random",f"searchspace-{args.searchspace}")
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

files = metasplits[f"train-{args.split}"].dropna().tolist()+metasplits[f"valid-{args.split}"].dropna().tolist()+metasplits[f"test-{args.split}"].dropna().tolist()

import copy
for file in files:
    response = normalized_dataset.global_surr[file][:,0]
    y = copy.deepcopy(response)
    rng.shuffle(y)
    results            = regret(y,response)
    results.to_csv(os.path.join(savedir,f"{file}.csv"))    