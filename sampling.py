#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:21:42 2020

@author: hsjomaa
"""

import tensorflow as tf
import pandas as pd
import random
import numpy as np
np.random.seed(318)
random.seed(3718)
tf.random.set_seed(0)
class Batch(object):
    
    def __init__(self,batch_size,fixed_shape = True):
        
        self.batch_size = batch_size
        self.fixed_shape = fixed_shape
        self.clear()
    
    def clear(self):
        # flattened triplets
        self.x = []
        # number of instances per item in triplets
        self.instances = []
        # number of features per item in triplets
        self.features = []
        # number of classes per item in triplets
        self.classes = []
        # model input
        self.input = None
        
    def append(self,instance):
        
        if len(self.x)==self.batch_size:
            
            self.clear()
            
        self.x.append(instance[0])
        self.instances.append(instance[1])
        self.features.append(instance[2])
        self.classes.append(instance[3])
        
    def collect(self):
        
        if len(self.x)!= self.batch_size and self.fixed_shape:
            raise(f'Batch formation incomplete!\n{len(self.x)}!={self.batch_size}')
        self.input = (tf.concat(self.x,axis=0),
                      tf.cast(tf.transpose(tf.concat(self.classes,axis=0)),dtype=tf.int32),
                      tf.cast(tf.transpose(tf.concat(self.features,axis=0)),dtype=tf.int32),
                      tf.cast(tf.transpose(tf.concat(self.instances,axis=0)),dtype=tf.int32),
                      )
        self.output = {'similaritytarget':tf.concat([tf.ones(self.batch_size),tf.zeros(self.batch_size)],axis=0)}

def pool(n,ntotal,shuffle):
    _pool = [_ for _ in list(range(ntotal)) if _!= n]
    if shuffle:
        random.shuffle(_pool)
    return _pool

class Sampling(object):
    def __init__(self,dataset):
        self.dataset          = dataset
        self.distribution     = pd.DataFrame(data=None,columns=['targetdataset','sourcedataset'])
        self.targetdataset   = None

    def sample(self,batch,split,sourcesplit):
        
        nsource  = len(self.dataset.orig_data[sourcesplit])
        ntarget  = len(self.dataset.orig_data[split])
        targetdataset = np.random.choice(ntarget,batch.batch_size)
        # clear batch
        batch.clear() 
        # find the negative dataset list of batch_size
        sourcedataset = []
        for target in targetdataset:
            if split==sourcesplit:
                swimmingpool  = pool(target,nsource,shuffle=True)  
            else:
                swimmingpool  = pool(-1,nsource,shuffle=True)
            sourcedataset.append(np.random.choice(swimmingpool))
        sourcedataset = np.asarray(sourcedataset).reshape(-1,)
        for target,source in zip(targetdataset,sourcedataset):
            # build instance
            instance = self.dataset.instances(target,source,split=split,sourcesplit=sourcesplit)
            batch.append(instance)
        
        distribution      = np.concatenate([targetdataset.reshape(-1,1),sourcedataset[:,None]],axis=1)
        self.distribution = pd.concat([self.distribution,\
                                       pd.DataFrame(distribution,columns=['targetdataset','sourcedataset'])],axis=0,ignore_index=True)
            
        self.targetdataset   = targetdataset  
        return batch
    
class TestSampling(object):
    def __init__(self,dataset):
        self.dataset          = dataset
        self.distribution     = pd.DataFrame(data=None,columns=['targetdataset','sourcedataset'])

    def sample(self,batch,split,sourcesplit,targetdataset):
        
        nsource  = len(self.dataset.orig_data[sourcesplit])
        # clear batch
        batch.clear() 
        # find the negative dataset list of batch_size
        swimmingpool  = pool(targetdataset,nsource,shuffle=True) if split==sourcesplit else pool(-1,nsource,shuffle=True)
        # double check divisibilty by batch size
        sourcedataset = np.random.choice(swimmingpool,batch.batch_size,replace=False)
        # iterate over batch negative datasets
        for source in sourcedataset:
            # build instance
            instance = self.dataset.instances(targetdataset,source,split=split,sourcesplit=sourcesplit)
            batch.append(instance)
        
        distribution      = np.concatenate([np.asarray(batch.batch_size*[targetdataset])[:,None],sourcedataset[:,None]],axis=1)
        self.distribution = pd.concat([self.distribution,\
                                       pd.DataFrame(distribution,columns=['targetdataset','sourcedataset'])],axis=0,ignore_index=True)    
            
        return batch

    def sample_from_one_dataset(self,batch):
        
        # clear batch
        batch.clear() 
        # iterate over batch negative datasets
        for _ in range(batch.batch_size):
            # build instance
            instance = self.dataset.instances()
            batch.append(instance)
        
        return batch
    