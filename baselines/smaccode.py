#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:36:47 2020

@author: hsjomaa
"""

import argparse
import os
import pandas as pd
import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.configspace import Configuration
from helper_fn import regret,warm_start
import sys
import json
np.random.seed(327)

parser = argparse.ArgumentParser()
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--k', help='number of starting points', type=int,default=20)
parser.add_argument('--index', help='Dataset index in the fold', type=int,default=0)
parser.add_argument('--method', help='Metafeatures used', choices=['aaai','tstr','d2v','random','transferable'],default='d2v')
parser.add_argument('--searchspace', type=str,default='a')

args = parser.parse_args()

currentdir     = os.path.dirname(os.path.realpath(__file__))
rootdir     = '/'.join(currentdir.split('/')[:-1])
sys.path.insert(0,rootdir)
from dataset import Dataset
savedir     = os.path.join(rootdir,"results","smac",f"init-{args.k}",args.method,f"split-{args.split}" if args.method=="d2v" else "")
outpdir     = os.path.join(rootdir,"outputs","smac",f"init-{args.k}",args.method,f"split-{args.split}" if args.method=="d2v" else "")
os.makedirs(savedir,exist_ok=True)
os.makedirs(outpdir,exist_ok=True)


splits_file       = os.path.join(rootdir, "metadataset",f"searchspace-{args.searchspace}",f"searchspace-{args.searchspace}-splits.csv")
metasplits        = pd.read_csv(splits_file,index_col=0)
info_file         = os.path.join(rootdir, "metadataset"  ,"info.json")
configuration     = json.load(open(info_file,'r'))[args.searchspace]
# specific to rgpe
configuration['split']                     = args.split
configuration['searchspace']               = args.searchspace
configuration['minmax']                    = True

normalized_dataset         = Dataset(configuration,rootdir,use_valid=True)
space_file                 = os.path.join(rootdir, "metadataset",f"searchspace-{args.searchspace}",f"searchspace-{args.searchspace}.json")
Configurations             = json.load(open(space_file,'r'))
Configurations             = [Configuration(normalized_dataset.get_cs(),_) for _ in list(Configurations.values())]
def config2idx(values):
    try:
        return Configurations.index(values)
    except:
        return None

def objective_fn(file):
    # minimize this objective
    response = normalized_dataset.global_surr[file][:,0]
    
    def obx(values):
        index    = config2idx(values)
        if index is not None:
            return 1. - response[index]
        else:
            return 1.
    return obx

metafeatures = {'aaai':pd.read_csv(os.path.join(rootdir,"metafeatures","mf1.csv"),index_col=0,header=0),
                    'tstr':pd.read_csv(os.path.join(rootdir,"metafeatures","mf2.csv"),index_col=0,header=0),
                    'd2v':pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{args.split}.csv"),index_col=0,header=0)}
    
files = metasplits[f"train-{args.split}"].dropna().tolist()+metasplits[f"valid-{args.split}"].dropna().tolist()+metasplits[f"test-{args.split}"].dropna().tolist()

file = files[args.index]

x        = warm_start(method=args.method,dataset=normalized_dataset,k=args.k,file=file,metafeatures=metafeatures)

response = normalized_dataset.global_surr[file][:,0]

# initiaize SMAC scenario
scenario = Scenario({"run_obj": "quality", "runcount-limit": args.n_iters, "cs": normalized_dataset.get_cs(),"deterministic": "true","output_dir":outpdir})
# initialize smace runner
initial_configurations = [Configuration(normalized_dataset.get_cs(),Configurations[_]) for _ in x]
# obtain configuration from idx
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(4),
                tae_runner=objective_fn(file),
                initial_configurations=initial_configurations,
                initial_design=None)
incumbent = smac.optimize()
output    = smac.runhistory.get_all_configs()
y = []
for _ in output:
    x = config2idx(_)
    if x is None:
        y.append(-1)
    else:
        y.append(response[x])
results = regret(y,response)
results.to_csv(os.path.join(savedir,f"{file}.csv"))
import shutil
shutil.rmtree(outpdir)