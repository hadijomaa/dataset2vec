#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:41:34 2021

@author: hsjomaa
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
np.random.seed(92)

def ptp(X):
    param_nos_to_extract = X.shape[1]
    domain = np.zeros((param_nos_to_extract, 2))
    for i in range(param_nos_to_extract):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])
    X = (X - domain[:, 0]) / np.ptp(domain, axis=1)
    return X

def flatten(x,y):
    '''
    Genearte x_i,y_j for all i,j \in |x|

    Parameters
    ----------
    x : numpy.array
        predictors; shape = (n,m)
    y : numpy.array
        targets; shape = (n,t)

    Returns
    -------
    numpy.array()
        shape ((n\times m\times t)\times 2).

    '''
    x_stack = []
    for c in range(y.shape[1]):
        c_label = np.tile(y[:,c],reps=[x.shape[1]]).transpose().reshape(-1,1)
        x_stack.append(np.concatenate([x.transpose().reshape(-1,1),c_label],axis=-1))
    return np.vstack(x_stack)

class Dataset(object):
    
    def __init__(self,file,rootdir):
            # read dataset
            self.X,self.y = self.__get_data(file,rootdir=rootdir)

            # batch properties
            self.ninstanc = 256
            self.nclasses = 3
            self.nfeature = 16
            
             
    def __get_data(self,file,rootdir):

        # read dataset folds
        datadir = os.path.join(rootdir, "datasets", file)
        # read internal predictors
        data = pd.read_csv(f"{datadir}/{file}_py.dat",header=None)
        # transform to numpy
        data    = np.asarray(data)
        # read internal target
        labels = pd.read_csv(f"{datadir}/labels_py.dat",header=None)
        # transform to numpy
        labels    = np.asarray(labels)        

        return data,labels

    def sample_batch(self,data,labels,ninstanc,nclasses,nfeature):
        '''
        Sample a batch from the dataset of size (ninstanc,nfeature)
        and a corresponding label of shape (ninstanc,nclasses).

        Parameters
        ----------
        data : numpy.array
            dataset; shape (N,F) with N >= nisntanc and F >= nfeature
        labels : numpy.array
            categorical labels; shape (N,) with N >= nisntanc
        ninstanc : int
            Number of instances in the output batch.
        nclasses : int
            Number of classes in the output label.
        nfeature : int
            Number of features in the output batch.

        Returns
        -------
        data : numpy.array
            subset of the original dataset.
        labels : numpy.array
            one-hot encoded label representation of the classes in the subset

        '''
        # Create the one-hot encoder
        ohc           = OneHotEncoder(categories = [range(len(np.unique(labels)))],sparse=False)
        d = {ni: indi for indi, ni in enumerate(np.unique(labels))}
        # process the labels
        labels        = np.asarray([d[ni] for ni in labels.reshape(-1)]).reshape(-1)
        # transform the labels to one-hot encoding
        labels        = ohc.fit_transform(labels.reshape(-1,1))
        # ninstance should be less than or equal to the dataset size
        ninstanc            = np.random.choice(np.arange(0,data.shape[0]),size=np.minimum(ninstanc,data.shape[0]),replace=False)
        # nfeature should be less than or equal to the dataset size
        nfeature         = np.random.choice(np.arange(0,data.shape[1]),size=np.minimum(nfeature,data.shape[1]),replace=False)
        # nclasses should be less than or equal to the total number of labels
        nclasses         = np.random.choice(np.arange(0,labels.shape[1]),size=np.minimum(nclasses,labels.shape[1]),replace=False)
        # extract data at selected instances
        data          = data[ninstanc]
        # extract labels at selected instances
        labels        = labels[ninstanc]
        # extract selected features from the data
        data          = data[:,nfeature]
        # extract selected labels from the data
        labels        = labels[:,nclasses]
        return data,labels
    
    def instances(self,ninstanc=None,nclasses=None,nfeature=None):
        # check if ninstance is provided
        ninstanc = ninstanc if ninstanc is not None else self.ninstanc
        # check if ninstance is provided
        nclasses = nclasses if nclasses is not None else self.nclasses
        # check if ninstance is provided
        nfeature = nfeature if nfeature is not None else self.nfeature        
        # check if neg batch is provided
        instance_x,instance_i = [],[]
        # append information to the placeholders
        x,y = self.sample_batch(self.X,self.y,ninstanc,nclasses,nfeature)
        instance_i.append(x.shape+(y.shape[1],)+(-1,))
        instance_x.append(flatten(x,y))
        # remove x,y
        del x,y
        # stack x values
        x = np.vstack(instance_x)
        # stack ninstanc
        ninstance = np.vstack(instance_i)[:,0][:,None]
        # stack nfeatures
        nfeature = np.vstack(instance_i)[:,1][:,None]
        # stack nclasses
        nclasses = np.vstack(instance_i)[:,2][:,None]
        # get task description of surr task
        return x,ninstance,nfeature,nclasses