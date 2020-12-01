#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:20:57 2020

@author: hsjomaa
"""

import numpy as np
import os
import pandas as pd
np.random.seed(826)

class Metadataset(object):
    def __init__(self, file,rootdir,searchspace,split):
        '''
        Initialize metadataset for file

        Parameters
        ----------
        file : str
            name of dataset.

        Returns
        -------
        None.

        '''
        # metadataset properties
        self.file = file
        self.split = split
        self.searchspace = searchspace
        self.rootdir = rootdir
        # get dataset folds (pre-defined)
        self.data,self.targets = self._get_data()
        # get metafeatures
        self._num_instances = {}
        self._num_instances.update({'train':self.data['train'].shape[0]})
        self._num_instances.update({'valid':self.data['valid'].shape[0]})
        self._num_instances.update({'test':self.data['test'].shape[0]})
        # get target space dimensionality
        self._num_targets       = len(np.unique(self.targets['train']))
        # get feature space cardinality
        self._num_predictors     = self.data['train'].shape[1]
        # get metadataset
        self.metadata     = self._get_metadata()
        # get metafeatures
        self.metafeatures = {}
        self.metafeatures.update({'d2v':self._get_metafeatures(specs='d2v')})
        self.metafeatures.update({'tstr':self._get_metafeatures(specs='tstr')})
        self.metafeatures.update({'aaai':self._get_metafeatures(specs='aaai')})
        
    def _get_metafeatures(self,specs):
        '''
        get pre-estimated meta-features

        Parameters
        ----------
        specs : str
            type of metafeatures.

        Returns
        -------
        None.

        '''
        if specs=='aaai':
            meatfeatures = pd.read_csv(os.path.join(self.rootdir,"metafeatures","mf1.csv"),index_col=0,header=0)
        elif specs=='tstr':
            meatfeatures = pd.read_csv(os.path.join(self.rootdir,"metafeatures","mf2.csv"),index_col=0,header=0)
        elif specs=='d2v':
            meatfeatures = pd.read_csv(os.path.join(self.rootdir,f"metafeatures",f"meta-features-split-{self.split}.csv"),index_col=0,header=0)
        metafeatures = meatfeatures.loc[self.file]
        return np.asarray(metafeatures).reshape(-1,)
    
    def _get_metadata(self):
        '''
        Get metadataset of internal file

        '''
        # read metadaaset file
        metadata = pd.read_csv(os.path.join(self.rootdir, "metadataset", f"searchspace-{self.searchspace}",f"{self.file}.txt"),index_col=None,header=None)
        return metadata
    
    def _get_data(self):
        '''
        Get internal dataset splits
        '''
        # read dataset folds
        datadir = os.path.join(self.rootdir, "datasets", self.file)
        folds = pd.read_csv(f"{datadir}/folds_py.dat",header=None)[0]
        # define internal fold
        folds = np.asarray(folds)
        # define validation fold
        vldfold = pd.read_csv(f"{datadir}/validation_folds_py.dat",header=None)[0]
        # get validation fold
        vldfold = np.asarray(vldfold)
        # read internal predictors
        predictors = pd.read_csv(f"{datadir}/{self.file}_py.dat",header=None)
        # transform to numpy
        predictors    = np.asarray(predictors)
        # get data folds
        data = {}
        data.update({'train':predictors[(1-folds)==1 & (vldfold==0)]})
        data.update({'test': predictors[folds==1]})
        data.update({'valid':predictors[vldfold==1]})
        # read internal target
        targets = pd.read_csv(f"{datadir}/labels_py.dat",header=None)
        # transform to numpy
        targets    = np.asarray(targets)        
        # get label folds
        labels = {}
        labels.update({'train':targets[(1-folds)==1 & (vldfold==0)]})
        labels.update({'test': targets[folds==1]})
        labels.update({'valid':targets[vldfold==1]})        

        return data,labels