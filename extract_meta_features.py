#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:34:17 2021

@author: hsjomaa
"""


import tensorflow as tf
import json
import numpy as np
import argparse
import os
from sampling import TestSampling,Batch
from dummdataset import Dataset
from modules import FunctionF,FunctionH,FunctionG,PoolF,PoolG
import pandas as pd
tf.random.set_seed(0)
np.random.seed(42)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--split', help='Select metafeature extraction model (one can take the average of the metafeatures across all 5 splits)', type=int,default=0)
parser.add_argument('--file', help='Select dataset name', type=str)

args    = parser.parse_args()
args.file = "abalone"

def Dataset2VecModel(configuration):

    nonlinearity_d2v  = configuration['nonlinearity_d2v']
    # Function F
    units_f     = configuration['units_f']
    nhidden_f   = configuration['nhidden_f']
    architecture_f = configuration['architecture_f']
    resblocks_f = configuration['resblocks_f']

    # Function G
    units_g     = configuration['units_g']
    nhidden_g   = configuration['nhidden_g']
    architecture_g = configuration['architecture_g']
    
    # Function H
    units_h     = configuration['units_h']
    nhidden_h   = configuration['nhidden_h']
    architecture_h = configuration['architecture_h']
    resblocks_h   = configuration['resblocks_h']
    #
    batch_size = configuration["batch_size"]
    trainable = False
    # input two dataset2vec shape = [None,2], i.e. flattened tabular batch
    x      = tf.keras.Input(shape=(2),dtype=tf.float32)
    # Number of sampled classes from triplets
    nclasses = tf.keras.Input(shape=(batch_size),dtype=tf.int32,batch_size=1)
    # Number of sampled features from triplets
    nfeature = tf.keras.Input(shape=(batch_size),dtype=tf.int32,batch_size=1)
    # Number of sampled instances from triplets
    ninstanc = tf.keras.Input(shape=(batch_size),dtype=tf.int32,batch_size=1)
    # Encode the predictor target relationship across all instances
    layer    = FunctionF(units = units_f,nhidden = nhidden_f,nonlinearity = nonlinearity_d2v,architecture=architecture_f,resblocks=resblocks_f,trainable=trainable)(x)
    # Average over instances
    layer    = PoolF(units=units_f)(layer,nclasses[0],nfeature[0],ninstanc[0])
    # Encode the interaction between features and classes across the latent space
    layer    = FunctionG(units = units_g,nhidden   = nhidden_g,nonlinearity = nonlinearity_d2v,architecture = architecture_g,trainable=trainable)(layer)
    # Average across all instances
    layer    = PoolG(units=units_g)(layer,nclasses[0],nfeature[0])
    # Extract the metafeatures
    metafeatures    = FunctionH(units = units_h,nhidden   = nhidden_h, nonlinearity = nonlinearity_d2v,architecture=architecture_h,trainable=trainable,resblocks=resblocks_h)(layer)
    # define hierarchical dataset representation model
    dataset2vec     = tf.keras.Model(inputs=[x,nclasses,nfeature,ninstanc], outputs=metafeatures)
    return dataset2vec


rootdir       = os.path.dirname(os.path.realpath(__file__))
log_dir       = os.path.join(rootdir,"checkpoints",f"split-{args.split}")
save_dir      = os.path.join(rootdir,"extracted")
configuration = json.load(open(os.path.join(log_dir,"configuration.txt"),"r"))
os.makedirs(save_dir,exist_ok=True)

metafeatures = pd.DataFrame(data=None)
datasetmf = []

batch       = Batch(configuration['batch_size'])
dataset     = Dataset(args.file,rootdir)
testsampler = TestSampling(dataset=dataset)


    
model     = Dataset2VecModel(configuration)

model.load_weights(os.path.join(log_dir,"weights"), by_name=False, skip_mismatch=False)

for q in range(10): # any number of samples
    batch = testsampler.sample_from_one_dataset(batch)
    batch.collect()
    datasetmf.append(model(batch.input).numpy())

metafeatures = np.vstack(datasetmf).mean(axis=0)[None]

pd.DataFrame(metafeatures,index=[args.file]).to_csv(os.path.join(save_dir,f"{args.file}.csv"))
