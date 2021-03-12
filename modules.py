#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:02:51 2020

@author: hsjomaa
"""

import tensorflow as tf

def importance(config):
    if config['importance'] == 'linear':
        fn = lambda x:x
    elif config['importance'] == 'None':
        fn = None
    else:
        raise('please define n importance function')
    return fn
    
ARCHITECTURES = ['SQU','ASC','DES','SYM','ENC']
def get_units(idx,neurons,architecture,layers=None):
    assert architecture in ARCHITECTURES
    if architecture == 'SQU':
        return neurons
    elif architecture == 'ASC':
        return (2**idx)*neurons
    elif architecture == 'DES':
        return (2**(layers-1-idx))*neurons    
    elif architecture=='SYM':
        assert (layers is not None and layers > 2)
        if layers%2==1:
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        
    elif architecture=='ENC':
        assert (layers is not None and layers > 2)
        if layers%2==0:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1            
            return neurons*2**(int(layers/2)-1 -idx)
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx
            return neurons*2**(int(layers/2) -idx)

class Function(tf.keras.layers.Layer):

    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):
        
        super(Function, self).__init__()
        
        self.n            = nhidden
        self.units        = units        
        self.nonlinearity = tf.keras.layers.Activation(nonlinearity)
        self.block        = [tf.keras.layers.Dense(units=get_units(_,self.units,architecture,self.n),trainable=trainable) \
                             for _ in range(self.n)]
            
    def call(self):
        raise Exception("Call not implemented")

class ResidualBlock(Function):

    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):

        super().__init__(units, nhidden, nonlinearity,architecture,trainable)
            
    def call(self, x):
        e = x + 0
        for i, layer in enumerate(self.block):
            e = layer(e)
            if i < (self.n - 1):
                e = self.nonlinearity(e)
        return self.nonlinearity(e + x)
        

class FunctionF(Function):
    
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable, resblocks=0):
        # m number of residual blocks
        super().__init__(units, nhidden, nonlinearity,architecture,trainable)
        # override function with residual blocks
        self.resblocks=resblocks
        if resblocks>0:
            self.block        = [tf.keras.layers.Dense(units=self.units,trainable=trainable)]
            assert(architecture=="SQU")
            self.block       += [ResidualBlock(units=self.units,architecture=architecture,nhidden=self.n,trainable=trainable,nonlinearity=self.nonlinearity) \
                                 for _ in range(resblocks)]                
            self.block       += [tf.keras.layers.Dense(units=self.units,trainable=trainable)]
        
    def call(self, x):
        e = x
        
        for i,fc in enumerate(self.block):
            
            e = fc(e)
            
            # make sure activation only applied once!
            if self.resblocks == 0:
                e = self.nonlinearity(e)
            else:
                # only first one
                if i==0 or i == (len(self.block)-1):
                    e = self.nonlinearity(e)    

        return e

class PoolF(tf.keras.layers.Layer):

    def __init__(self,units):    
        super(PoolF, self).__init__()
        
        self.units = units
        
    def call(self,x,nclasses,nfeature,ninstanc):
        
        s = tf.multiply(nclasses,tf.multiply(nfeature,ninstanc))
        
        x           = tf.split(x,num_or_size_splits=s,axis=0)
        
        e  = []
        
        for i,bx in enumerate(x):
            
            te     = tf.reshape(bx,shape=(1,nclasses[i],nfeature[i],ninstanc[i],self.units))
            
            te     = tf.reduce_mean(te,axis=3)
            e.append(tf.reshape(te,shape=(nclasses[i]*nfeature[i],self.units)))
            
        e = tf.concat(e,axis=0)
        
        return e
    
class FunctionG(Function):
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):
        
        super().__init__(units, nhidden, nonlinearity,architecture,trainable)
    def call(self, x):
        e = x
        
        for fc in self.block:
            
            e = fc(e)
            
            e = self.nonlinearity(e)

        return e

class PoolG(tf.keras.layers.Layer):

    def __init__(self,units):    
        super(PoolG, self).__init__()
        
        self.units = units
        
    def call(self, x,nclasses,nfeature):
        
        s = tf.multiply(nclasses, nfeature)      
        
        x = tf.split(x,num_or_size_splits=s,axis=0)
        
        e  = []
        
        for i,bx in enumerate(x):
            
            te     = tf.reshape(bx,shape=(1,nclasses[i]*nfeature[i],self.units))
            
            te     = tf.reduce_mean(te,axis=1)
            
            e.append(te)
            
        e = tf.concat(e,axis=0)

        return e
    

class FunctionH(Function):
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable, resblocks=0):
        # m number of residual blocks
        super().__init__(units, nhidden, nonlinearity,architecture,trainable)
        # override function with residual blocks
        self.resblocks = resblocks
        if resblocks>0:
            self.block        = [tf.keras.layers.Dense(units=self.units,trainable=trainable)]
            assert(architecture=="SQU")
            self.block       += [ResidualBlock(units=self.units,architecture=architecture,nhidden=self.n,trainable=trainable,nonlinearity=self.nonlinearity) \
                                 for _ in range(resblocks)]                
            self.block       += [tf.keras.layers.Dense(units=self.units,trainable=trainable)]
        
    def call(self,x):
        
        e = x
        
        for i,fc in enumerate(self.block):
            
            e = fc(e)
            # make sure activation only applied once!
            if self.resblocks == 0:
                if i<(len(self.blocks)-1):
                    e = self.nonlinearity(e)
            else:
                # only first one
                if i==0:
                    e = self.nonlinearity(e)      

        return e


class PoolH(tf.keras.layers.Layer):
    def __init__(self, batch_size,units):
        """
        """
        super(PoolH, self).__init__()
        self.batch_size = batch_size
        self.units = units
        
    def call(self, x,ignore_negative):
        
        e  =  tf.reshape(x,shape=(self.batch_size,3,self.units))
        # average positive meta-features
        e1 = tf.reduce_mean(e[:,:2],axis=1)
        if not ignore_negative:
            # select negative meta-feautures 
            e1 = e[:,-1][:,None]            
        # reshape, i.e. output is [batch_size,nhidden]
        e  = tf.reshape(e1,shape=(self.batch_size,self.units))            
        
        return e