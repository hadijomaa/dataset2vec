#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:40:44 2020

@author: hsjomaa
"""

import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import numpy as np
import json
import pandas as pd
import os
import sys
from scipy.stats import norm
from helper_fn import regret,warm_start

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

np.random.seed(327)

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', help='Metafeatures used', choices=['32','52','rbf'],default='rbf')
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


def propose(optimal, gprm,indices):
    mu, sigma     = gprm.mean().numpy(),gprm.stddev().numpy()
    mu            = mu.reshape(-1,)
    sigma         = sigma.reshape(-1,)
    with np.errstate(divide='warn'):
        imp = mu - optimal
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    for _ in range(len(ei)):
        if _ in indices:
            ei[_] = 0
    return np.argmax(ei)

savedir     = os.path.join(rootdir,"results",f"gaussian-kernel-{args.kernel}",f"searchspace-{args.searchspace}",f"init-{args.k}",args.method,f"split-{args.split}" if args.method=="d2v" else "")
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

kernels = {'32':tfk.MaternThreeHalves,
           '52':tfk.MaternFiveHalves,
           'rbf':tfk.ExponentiatedQuadratic}

def build_gp(x,amplitude, length_scale, z):
  kernel = tfk.FeatureScaled(kernels[args.kernel](amplitude),scale_diag=length_scale)
  return tfd.GaussianProcess(kernel=kernel,index_points=x,observation_noise_variance=z)
  
class GaussianProcess(object):
    # create Gaussian Process model for linear regression
    def __init__(self,X,Y):
        # data
        self.X   = X
        self.Y   = Y
        self.get_trainable_variables()
        self.gp = build_gp(self.X,self.amplitude_var,self.length_scale_var,self.observation_noise_variance_var)
    
    def sample(self):
        '''
        Sample from the Joint Gaussian Distribution

        Returns
        -------
        x: tf.Tensor
            Sample of the process.

        '''
        x = self.gp.sample()
        
        return x
    
    def log_prob(self,x):
        '''
        Estimate the log probability of the a sample value

        Parameters
        ----------
        x : tf.Tensor
            DESCRIPTION.

        Returns
        -------
        lp.

        '''
        lp = self.gp.log_prob(x)
        return lp

    def get_trainable_variables(self):
        constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())
        
        self.amplitude_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='amplitude',
            dtype=np.float64)
        
        self.length_scale_var = tfp.util.TransformedVariable(
            initial_value=np.ones(self.X.shape[1],dtype=np.float64),
            bijector=constrain_positive,
            name='length_scale',
            dtype=np.float64)
        
        self.observation_noise_variance_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='observation_noise_variance_var',
            dtype=np.float64)
        
        self.trainable_variables = [v.trainable_variables[0] for v in 
                                [self.amplitude_var,
                                self.length_scale_var,
                                self.observation_noise_variance_var]]
    # Use `tf.function` to trace the loss for more efficient evaluation.
    @tf.function(autograph=False, experimental_compile=False)
    def target_log_prob(self,y):
      return self.gp.log_prob(y)
  
data_dict   = {}


metafeatures = {'aaai':pd.read_csv(os.path.join(rootdir,"metafeatures","mf1.csv"),index_col=0,header=0),
                    'tstr':pd.read_csv(os.path.join(rootdir,"metafeatures","mf2.csv"),index_col=0,header=0),
                    'd2v':pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{args.split}.csv"),index_col=0,header=0)}
    
files = metasplits[f"train-{args.split}"].dropna().tolist()+metasplits[f"valid-{args.split}"].dropna().tolist()+metasplits[f"test-{args.split}"].dropna().tolist()

file = files[args.index]

x        = warm_start(method=args.method,dataset=normalized_dataset,k=args.k,file=file,metafeatures=metafeatures)

response = normalized_dataset.global_surr[file][:,0]
  
y = np.vstack([response[_] for _ in x]).reshape(-1,)
  
q = np.vstack([normalized_dataset.orighyper[_] for _ in x])

x = list(x)

opt = max(response)
for _ in range(args.n_iters):
    sol = [normalized_dataset.global_surr[file][_,0] for _ in x]
    if opt in sol:
        break        
    # define a new model
    model     = GaussianProcess(X=q,Y=y)
    optimizer = tf.optimizers.Adam(learning_rate=.01)

    losses          = []
    bestloss        = np.inf
    earlystopping = 0
    while earlystopping < 16:
        with tf.GradientTape() as tape:
          loss = -model.target_log_prob(y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if abs(bestloss-loss.numpy()) < 1e-3:
            earlystopping +=1
        else:
            earlystopping = 0
            bestloss      = loss.numpy()
        losses.append(loss)
              
    optimized_kernel = tfk.FeatureScaled(kernels[args.kernel](model.amplitude_var),scale_diag=model.length_scale_var)
    gprm             = tfd.GaussianProcessRegressionModel(kernel=optimized_kernel,index_points=normalized_dataset.orighyper,observation_index_points=q,
                                                          observations=y,
                                                          observation_noise_variance=model.observation_noise_variance_var,
                                                          predictive_noise_variance=0.,)
    
    ymax      = max(y)
    candidate = propose(ymax,gprm,x)
    x.append(candidate)
    q = np.vstack([normalized_dataset.orighyper[_] for _ in x])
    y = np.vstack([response[_] for _ in x]).reshape(-1,)
results            = regret(y,response)
results.to_csv(os.path.join(savedir,f"{file}.csv"))