#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:34:23 2020

@author: hsjomaa
"""

import argparse
import os
import pandas as pd
import numpy as np

# get meta-split as argument
parser = argparse.ArgumentParser()
parser.add_argument('--split', help='Select training fold', type=int,default=4)
args = parser.parse_args()

rootdir     = os.path.dirname(os.path.realpath(__file__))

def ptp(X):
    param_nos_to_extract = X.shape[1]
    domain = np.zeros((param_nos_to_extract, 2))
    for i in range(param_nos_to_extract):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])
    return domain

x = pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-feautures-split-{args.split}-unprocessed.csv"),header=0,index_col=0)

domain = ptp(np.asarray(x))

normalize = lambda X : (X - domain[:, 0]) / np.ptp(domain, axis=1)

indices = x.index.tolist()
x = normalize(x)
pd.DataFrame(x,index=indices).fillna(0).to_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{args.split}.csv"))