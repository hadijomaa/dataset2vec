#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:58:58 2020

@author: hsjomaa
"""

import tensorflow as tf
import pandas as pd
import json
import time
import os
from modules import FunctionF,FunctionH,FunctionG,PoolF,PoolG,PoolH
tf.random.set_seed(0)


class Model(object):
    '''
    Model
    '''
    def __init__(self,configuration,rootdir,for_eval=False,fine_tuning=False):

        # data shape
        self.batch_size    = configuration['batch_size']
        self.split         = configuration['split']
        self.searchspace   = configuration['searchspace']
        self.split         = configuration['split']
        
        
        self.nonlinearity_d2v  = configuration['nonlinearity_d2v']
        # Function F
        self.units_f     = configuration['units_f']
        self.nhidden_f   = configuration['nhidden_f']
        self.architecture_f = configuration['architecture_f']
        self.resblocks_f = configuration['resblocks_f']

        # Function G
        self.units_g     = configuration['units_g']
        self.nhidden_g   = configuration['nhidden_g']
        self.architecture_g = configuration['architecture_g']
        
        # Function H
        self.units_h     = configuration['units_h']
        self.nhidden_h   = configuration['nhidden_h']
        self.architecture_h = configuration['architecture_h']
        self.resblocks_h   = configuration['resblocks_h']

        self.delta = configuration['delta']
        self.gamma = configuration['gamma']

        self.config_num = configuration["number"]
        
        self.model =self.dataset2vecmodel(trainable=True)
        self.trainable_count = int(sum([tf.keras.backend.count_params(p) for p in self.model.trainable_weights]))
        configuration["trainable"] = self.trainable_count

        # tracking
        self.metrickeys = ['similarityloss','time',"roc"]
        self.with_csv   = True
        # create a location if not evaluation model
        if not for_eval:
            self._create_metrics()
            self.directory = self._create_dir(rootdir)
            self._save_configuration(configuration)
            
    def _create_dir(self,rootdir):
        import datetime
        # create directory
        directory = os.path.join(rootdir, "checkpoints",f"searchspace-{self.searchspace}",f"split-{self.split}","dataset2vec",\
                                 "vanilla",f"configuration-{self.config_num}",datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
        os.makedirs(directory)
        return directory
    
    @tf.function
    def similarity(self,layer,positive_pair):
        '''

        Parameters
        ----------
        layer : tf.Tensor
            Extracted metafeatures; shape = [None,3,units_hh].
        positive_pair : bool
            indicator of similarity expected (between positive pair or negative pair).

        Returns
        -------
        tf.Tensor
            Similarity between metafeatures.

        '''
        # check if requires reshape
        return tf.exp(-self.gamma*self.distance(layer,positive_pair))
    
    @tf.function
    def distance(self,layer,positive_pair):
        '''
        Return the cosine similarity between dataset metafeatures

        Parameters
        ----------
        layer : tf.Tensor
            metafeatures.

        Returns
        -------
        cos : tf.Tensor
            Cosine similarity.

        '''
        # reshape metafeatures
        layer   = tf.reshape(layer,shape=(self.batch_size,3,self.units_h))
        # average metafeatures of positive data
        pos = tf.reduce_mean(layer[:,:2],axis=1)[:,None] if not positive_pair else layer[:,0][:,None]
        # metafeatures of negative data
        neg = layer[:,-1][:,None] if not positive_pair else layer[:,1][:,None]                   
        # concatenate with negative meta-features
        layer  = tf.keras.layers.concatenate([pos,neg],axis=1)
        dist = tf.norm(layer[:,0]-layer[:,1],axis=1)
        return dist
    
    @tf.function
    def similarityloss(self,target_y,predicted_y):
        '''
        Compute the similarity log_loss between positive-pair metafeatures
        and negative-pair metafeatures.

        Parameters
        ----------
        target_y : tf.Tensor
            Similarity indicator.
        predicted_y : tf.Tensor
            Extracted metafeatures; shape = [None,3,units_hh].

        Returns
        -------
        tf.Tensor

        '''
        negative_prob   = self.similarity(predicted_y,positive_pair=False)
        
        positive_prob   = self.similarity(predicted_y,positive_pair=True)
        
        logits          = tf.concat([positive_prob,negative_prob],axis=0)
        # create weight
        similarityweight = tf.concat([tf.ones(shape=self.batch_size),self.delta*(tf.ones(shape=self.batch_size))],axis=0)
        
        return tf.compat.v1.losses.log_loss(labels=target_y,predictions=logits, weights=similarityweight)
    
    @tf.function
    def loss(self,target_y,output,training=True):
        '''
        Compute the total loss of the network.

        Parameters
        ----------
        target_y : tuple(tf.Tensor)
            (Similarty Indicator,targetresponse)
        output : tuple
            Output of the model.
        training : bool
            important to specify keys of metrics dict.
        Returns
        -------
        loss : tf.Tensor

        '''
        # add prefix
        prefix = '' if training else 'vld'
        # parse target output
        similaritytarget = target_y['similaritytarget']
        # create metrics placeholder
        metrics = {}
        # split the output 
        metafeatures    = output['metafeatures']
        
        losses  = {}
        l       = None
        # Compute Similarity Loss
        losses.update({'similarity':self.similarityloss(similaritytarget,predicted_y=metafeatures)})
        l = losses['similarity']
        metrics.update({f'{prefix}similarityloss':losses['similarity']})
        return l,metrics
       
    def train_step(self,x,y,optimizer,clip=True,no_metrics=False):
        starttime = time.time()
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            output = self.model(x, training=True)
            loss,metrics = self.loss(y, output)
                
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # clip gradients
        if clip:
            gradients = [tf.clip_by_value(t,clip_value_min=-0.5,clip_value_max=0.5) for t in gradients]
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables ))
        # add time required to run backprop
        metrics.update({'time':tf.constant(time.time()-starttime)})
        # update metrics
        if no_metrics:
            pass
        else:
            self._update_metrics(metrics)
            # return metrics
            return metrics

    def update_tracker(self,training,metrics=None):
        '''
        update csv tracker

        Parameters
        ----------
        training : bool
            which tracker to update.
        metrics : dict
            dictionary of tf.keras.metrics, default None
        Returns
        -------
        None.

        '''
        # use existing metric or passed down ?
        metrics = self.metrics if metrics is None else metrics
        key = 'train' if training else 'valid'
        # iterate over metrics
        for metric,_ in metrics.items():
            # update cell in dataframe
            value  = self.metrics[metric]
            if metric != 'time':
                # check if passed down
                if hasattr(value,'result'):
                    value = value.result().numpy()
                elif hasattr(value,'numpy'):
                    value = value.numpy()
                else:
                    value = value
                self.csv[key].at[self.update_counter[key],metric] = value
            else:
                self.csv[key].at[self.update_counter[key],metric] = self.metrics[metric].result().numpy()
        # update key counter
        self.update_counter[key] += 1
        
    def _update_metrics(self,metrics):
        '''
        Update metrics trackers

        Parameters
        ----------
        metrics : dict

        Returns
        -------
        None.

        '''
        # iterate over metrics
        for metric,value in metrics.items():
            # update dictionaries
            # check if metric in keys
            if metric in list(self.metrics.keys()):
                # check if metric is a tf.keras  function
                if hasattr(self.metrics[metric],'call'):
                    # update metric
                    self.metrics[metric](value)
                # if the metric is not
                else:
                    self.metrics[metric] = value
            else:
                # add dictionary
                self.metrics.update({metric:value})
        
    def reset_states(self):
        '''
        Reset tracking metrics

        Returns
        -------
        None.

        '''
        for metric in self.metrics.keys():
            # check if metric is tf.keras function
            if hasattr(self.metrics[metric],'reset_states'):
                # do not reset time 
                if metric != 'time':
                    self.metrics[metric].reset_states()
            else:
                self.metrics[metric] = None
            
    def dump(self):
        '''
        Save csv progress
        '''
        for key in ['train','valid']:
            self.csv[key].to_csv(f'{self.directory}/{key}-progress.csv')
        
    def _create_metrics(self):
        '''
        Create tracking metrics

        Returns
        -------
        None.

        '''
        # create epoch counter
        self.update_counter = {'train':0,'valid':0}
        # create empty dictionary
        self.metrics = {}
        # fill dictionary with keys and values
        [self.metrics.update({_:tf.keras.metrics.Mean(name=_)}) for _ in self.metrickeys if _ !='time']
        # fix time metrics
        self.metrics['time'] = tf.keras.metrics.Sum(name='time')
        # check if csv required
        if self.with_csv:
            # create csv dictionary
            self.csv = {}
            # add training csv tracker
            self.csv.update({'train': pd.DataFrame(data=None,columns=[_ for _ in self.metrickeys if 'vld' not in _])})
            # add training csv tracker
            self.csv.update({'valid': pd.DataFrame(data=None,columns=[_ for _ in self.metrickeys if 'vld' in _])})
            
    def report(self):
        template    = 'Similarity: {:.5f}, ROC: {:.5f}, Time: {:.2f} s '
        print(template.format(self.metrics['similarityloss'].result(),
                              self.metrics['roc'].result(),
                               self.metrics['time'].result()))
        
    def _save_configuration(self,configuration):
        configuration.update({"savedir":self.directory})
        filepath = os.path.join(self.directory,"configuration.txt")
        with open(filepath, 'w') as json_file:
          json.dump(configuration, json_file)        
        
    def save_weights(self,iteration=None):
        '''
        Save weights of model with provided description
        
        Parameters
        ----------
        description: str
            name of weights to save.
        Returns
        -------
        None.

        '''
        # define filepath
        iteration = f"-{iteration}" if iteration is not None else ''
        filepath = os.path.join(self.directory,f"iteration{iteration}","weights")
        os.makedirs(filepath,exist_ok=True)
        # save internal model weights
        self.model.save_weights(filepath=os.path.join(filepath,"weights"))
             
    def set_weights(self,weights=None):
        '''
        Update the weights of the internal model with backend model
        weights or with provided weights.
        
        Parameters
        ----------
        weights : List[tf.Variable], optional
            Weights of the trainable variables. The default is None.

        '''
        self.model.set_weights(weights=weights)
    
    def get_weights(self,internal=True):
        '''
        Return weights of the (internal) model

        Parameters
        ----------
        internal : bool, optional
            indicator of type of model for which we want 
            to get weights. The default is True.

        Returns
        -------
        weights : list(tf.Tensor)

        '''
        # get weights
        weights = self.model.get_weights()
        return weights
    
    def getmetafeatures(self,x):
        
        output = self.model(x,training=False)
        
        layer    = PoolH(self.batch_size,self.units_h)(output['metafeatures'],ignore_negative=True)        
        
        return layer
    
    # @tf.function
    def predict(self,x,y):
        '''
        Return the distribution of the target task

        Parameters
        ----------
        x : tuple(tf.Tensor)
            input.
        y : tuple(tf.Tensor)
            output.
        Returns
        -------
        y_mean : TYPE
            DESCRIPTION.
        y_logvar : TYPE
            DESCRIPTION.

        '''
        # predict
        output = self.model(x,training=False)
        phi     = output['metafeatures']
        posprob = self.similarity(phi,positive_pair=True)
        negprob = self.similarity(phi,positive_pair=False)        
        proba  = tf.concat([posprob,negprob],axis=0)
        return proba,y["similaritytarget"]
        
    def dataset2vecmodel(self,trainable):
        # input two dataset2vec shape = [None,2], i.e. flattened tabular batch
        x      = tf.keras.Input(shape=(2),dtype=tf.float32)
        # Number of sampled classes from triplets
        nclasses = tf.keras.Input(shape=(self.batch_size*3),dtype=tf.int32,batch_size=1)
        # Number of sampled features from triplets
        nfeature = tf.keras.Input(shape=(self.batch_size*3),dtype=tf.int32,batch_size=1)
        # Number of sampled instances from triplets
        ninstanc = tf.keras.Input(shape=(self.batch_size*3),dtype=tf.int32,batch_size=1)
        # Encode the predictor target relationship across all instances
        layer    = FunctionF(units = self.units_f,nhidden = self.nhidden_f,nonlinearity = self.nonlinearity_d2v,architecture=self.architecture_f,resblocks=self.resblocks_f,trainable=trainable)(x)
        # Average over instances
        layer    = PoolF(units=self.units_f)(layer,nclasses[0],nfeature[0],ninstanc[0])
        # Encode the interaction between features and classes across the latent space
        layer    = FunctionG(units = self.units_g,nhidden   = self.nhidden_g,nonlinearity = self.nonlinearity_d2v,architecture = self.architecture_g,trainable=trainable)(layer)
        # Average across all instances
        layer    = PoolG(units=self.units_g)(layer,nclasses[0],nfeature[0])
        # Extract the metafeatures
        metafeatures    = FunctionH(units = self.units_h,nhidden   = self.nhidden_h, nonlinearity = self.nonlinearity_d2v,architecture=self.architecture_h,trainable=trainable,resblocks=self.resblocks_h)(layer)
        # define hierarchical dataset representation model
        output = {'metafeatures':metafeatures}
        dataset2vec     = tf.keras.Model(inputs=[x,nclasses,nfeature,ninstanc], outputs=output)
        return dataset2vec