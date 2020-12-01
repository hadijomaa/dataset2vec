#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:16:00 2020

@author: hsjomaa
"""

from scipy.stats import norm
import numpy as np
from pybnn.bohamiann import Bohamiann
import argparse
import json
import pandas as pd
import os
import sys
from helper_fn import regret,warm_start

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
rootdir   = '/'.join(currentdir.split('/')[:-1])
sys.path.insert(0,rootdir)
from dataset import Dataset

def propose(optimal, algorithm,indices):
    optimal       = -1*optimal
    mu, sigma     = algorithm.predict(normalized_dataset.orighyper)
    mu            = -1.*mu.reshape(-1,)
    sigma         = sigma.reshape(-1,)
    with np.errstate(divide='warn'):
        imp = mu - optimal
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    for _ in range(len(ei)):
        if _ in indices:
            ei[_] = 0
    return np.argmax(ei)

savedir     = os.path.join(rootdir,"results","bohamiann",f"searchspace-{args.searchspace}",f"init-{args.k}",args.method,f"split-{args.split}" if args.method=="d2v" else "")
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

file = files[args.index]

x        = warm_start(method=args.method,dataset=normalized_dataset,k=args.k,file=file,metafeatures=metafeatures)

response = 1 - normalized_dataset.global_surr[file][:,0]
    
y = np.vstack([response[_] for _ in x]).reshape(-1,)
  
q = np.vstack([normalized_dataset.orighyper[_] for _ in x])

x = list(x)

opt   = min(response)

output = []
for _ in range(args.n_iters):
    if opt in y:
        break    
    model = Bohamiann(print_every_n_steps=100,normalize_input=False)
    model.train(x_train=q, y_train=y,verbose=False)
    candidate   =     propose(opt,algorithm=model,indices=x)
    
    x.append(candidate)
    q = np.vstack([normalized_dataset.orighyper[_] for _ in x])
    y = np.vstack([response[_] for _ in x]).reshape(-1,)

response = normalized_dataset.global_surr[file][:,0]
y = np.vstack([response[_] for _ in x]).reshape(-1,)

results            = regret(y,response)
results.to_csv(os.path.join(savedir,f"{file}.csv"))