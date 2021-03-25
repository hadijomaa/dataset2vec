#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:19:28 2020

@author: hsjomaa
"""

import pandas as pd
import numpy as np
from metadataset import Metadataset
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.preprocessing import MinMaxScaler
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
    
    def __init__(self,configuration,rootdir,use_valid=False):
        # dataset properties
        self.split = configuration['split']
        # batch properties
        self.ninstanc = configuration['ninstanc']
        self.nclasses = configuration['nclasses']
        self.nfeature = configuration['nfeature']
        self.searchspace   = configuration['searchspace']
        self.minmax        = configuration['minmax']
        self.rootdir       = rootdir
        self.cardinality   = configuration['cardinality']
        self.D             = configuration['D']
        # Data dictionaries
        self.orig_data = {}
        # Label dictionaries
        self.orig_labl = {}
        # Surrogate dictionaries
        self.orig_surr = {}
        # metafeatures
        self.orig_d2v = {}            
        # metafeatures
        self.orig_tstr = {}   
        self.global_surr          = {}
        # metafeatures
        self.orig_aaai = {}
        self.orig_files = {}
        # create hyperparameter space
        self.orighyper    = None
        # read meta-splits
        splits_file = os.path.join(rootdir, "metadataset",f"searchspace-{configuration['searchspace']}",f"searchspace-{configuration['searchspace']}-splits.csv")
        metasplits  = pd.read_csv(splits_file,index_col=0)
        # iterate over splits of the meta-splits
        iterates = ['train','valid','test'] if use_valid else ['train','test']
        for split in iterates:
            # temporary data list
            data   = []
            # temporary label list
            labl   = []
            # temporary surrogate list
            surr   = []
            d2v = []
            aaai = []
            tstr = []
            # iterate over datasets in split
            # dataset names
            if use_valid:
                files = metasplits[f'{split}-{self.split}'].dropna().tolist()
            else:
                files = metasplits[f'{split}-{self.split}'].dropna().tolist() if split=='test'\
                else metasplits[f'{split}-{self.split}'].dropna().tolist() + metasplits[f'valid-{self.split}'].dropna().tolist()
            for c,file in enumerate(files):
                # read dataset
                dataset = Metadataset(file,rootdir=rootdir,searchspace=configuration["searchspace"],split=self.split)
                # append data to list
                data.append(dataset.data)
                # append labels to list
                labl.append(dataset.targets)
                # append metafeatures
                d2v.append(dataset.metafeatures['d2v'])
                aaai.append(dataset.metafeatures['aaai'])
                tstr.append(dataset.metafeatures['tstr'])
                # apply transformations
                dataset.metadata = np.asarray(dataset.metadata)
                # encode dataset.trn_tasks
                if self.minmax:
                    # create scaler
                    scaler = MinMaxScaler()
                    # fit transform data [0 - 1]
                    values = np.around(scaler.fit_transform(dataset.metadata[:,-1][:,None]),4)
                else:
                    values = dataset.metadata[:,-1][:,None]
                self.orighyper = ptp(dataset.metadata[:,:-1]) if self.orighyper is None else self.orighyper
                task =  np.concatenate([values,self.orighyper],axis=1)     
                # append surrogate to list
                surr.append(task)
                self.global_surr.update({file:task})
            self.orig_files.update({split:files})
            # add dataset of the split list to dictionary
            self.orig_data.update({split:data})
            # add labels of the split list to dictionary
            self.orig_labl.update({split:labl})
            # add surrogate of the split list to dictionary
            self.orig_surr.update({split:surr})
            # add metafeatures of the split list to dictionary
            self.orig_d2v.update({split:d2v})
            self.orig_tstr.update({split:tstr})
            self.orig_aaai.update({split:aaai})
            
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

    def _instance(self,targetdataset,split,fold,**kwags):
        data    = self.orig_data[split]
        # select labels list
        labels  = self.orig_labl[split]
        # sample batch from the train-split of the pos data
        x,y = self.sample_batch(data[targetdataset][fold],labels[targetdataset][fold],**kwags)        
        return x,y
    
    def instances(self,targetdataset,sourcedataset,split,ninstanc=None,nclasses=None,nfeature=None,sourcesplit=None):
        '''
        Build an instance for the model

        Parameters
        ----------
        pos : int
            index of the dataset for positive pair.
        neg : int
            index of the dataset for negative pair.
        surr : int
            index of the surrogate task.
        split : int
            represents the split from which the positive batch is 
            sampled, i.e. train test or validation.
        ninstanc : int
            Number of instances in the output batch.
        nclasses : int
            Number of classes in the output label.
        nfeature : int
            Number of features in the output batch.
        sourcesplit : int
            represents the split from which the negative batch is 
            sampled, i.e. train test or validation.

        Returns
        -------
        None.

        '''
        # check if ninstance is provided
        ninstanc = ninstanc if ninstanc is not None else self.ninstanc
        # check if ninstance is provided
        nclasses = nclasses if nclasses is not None else self.nclasses
        # check if ninstance is provided
        nfeature = nfeature if nfeature is not None else self.nfeature        
        # check if neg batch is provided
        sourcesplit = sourcesplit if sourcesplit is not None else split               
        # prepare placeholders
        instance_x,instance_i = [],[]
        # append information to the placeholders
        x,y = self._instance(targetdataset=targetdataset,split=split,fold='train',
                                      ninstanc=ninstanc,nclasses=nclasses,nfeature=nfeature)
        instance_i.append(x.shape+(y.shape[1],)+(targetdataset,))
        instance_x.append(flatten(x,y))
        # remove x,y
        del x,y
        # sample batch from the valid-split of the pos data
        x,y = self._instance(targetdataset=targetdataset,split=split,fold='valid',
                                      ninstanc=ninstanc,nclasses=nclasses,nfeature=nfeature)
        # append information to the placeholders
        instance_i.append(x.shape+(y.shape[1],)+(targetdataset,))
        instance_x.append(flatten(x,y))
        # remove x,y
        del x,y
        # sample batch from the train-split of the neg data
        x,y = self._instance(targetdataset=sourcedataset,split=sourcesplit,fold='train',
                                      ninstanc=ninstanc,nclasses=nclasses,nfeature=nfeature)
        # append information to the placeholders
        instance_i.append(x.shape+(y.shape[1],)+(sourcedataset,))
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

    def get_cs(self):
        '''
        Get configuration space

        Returns
        -------
        cs : ConfigSpace.ConfigurationSpace
            the configuration space.

        '''
        # import specific modules
        import ConfigSpace
        import json
        # create configuration space placeholder
        cs = ConfigSpace.ConfigurationSpace()
        # read associated json file
        space_file = os.path.join(self.rootdir, "metadataset",f"searchspace-{self.searchspace}",f"searchspace-{self.searchspace}-configurationspace.txt")
        configuration = json.load(open(space_file,'r'))
        for config in configuration.keys():
            val = configuration[config]
            assert(len(val)>1)
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(config, val))
        return cs
    