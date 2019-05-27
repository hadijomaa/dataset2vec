#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:27:11 2019

@author: hsjomaa
"""

import tensorflow as tf
import time
import datetime
import copy
import numpy as np
import os
import pickle
class Runner:

    def __init__(self, config, dataset, model):
        self.config = config
        self.ds     = dataset
        self.model  = model
        self.saver  = tf.train.Saver(max_to_keep=2)

    def run(self,checkpoint_dir=None,dataset=''):
        if checkpoint_dir is None:
            checkpoint_dir = datetime.datetime.now().strftime('./checkpoints/{}-%Y-%m-%d-%H-%M-%S-%f{}'.format(self.config['split'],dataset))
        os.makedirs(checkpoint_dir,exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        with open(os.path.join(checkpoint_dir,'config.p'), 'wb') as fp:
            pickle.dump(self.config, fp, protocol=pickle.HIGHEST_PROTOCOL)        
        with tf.Session() as sess:

            start_time = time.time()
            self.initialize_model(sess=sess)
            log_freq = self.config["performance_epoch_frequency"]
            loss_avg = 0;pos_loss_avg=0;neg_loss_avg = 0;
            grad_norm_pred_unclipped_avg, grad_norm_pred_clipped_avg = 0, 0

            print('NUMBER OF PARAMETERS: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.model.pred_variables])))
            print('{} | {} | {} | {} | {} | {} | {}'.format('EPOCH','LOSS','POSLOSS','NEGLOSS',
                            'GRADNORMUNCLIPPSED', 'GRADNORMCLIPPED','TIME'))
            best_loss = 1e9
            for epoch_idx in range(self.config["num_epochs"]):

                for k in range(self.config["steps_prediction"]):
                    X, I, num_pos = self.ds.get_batch(self.config["batch_size"])
                    L_hat_batch,pos_loss_batch,neg_loss_batch, grad_norm_pred_unclipped_batch, grad_norm_pred_clipped_batch = \
                            self.model.update_prediction_model(sess=sess, X=X,
                                                               N_features  = I[:,1],
                                                               N_classes   = I[:,2],
                                                               N_instances = I[:,0],
                                                               N_pos           = num_pos)
                    loss_avg += L_hat_batch
                    pos_loss_avg += pos_loss_batch
                    neg_loss_avg += neg_loss_batch
                    grad_norm_pred_unclipped_avg += grad_norm_pred_unclipped_batch
                    grad_norm_pred_clipped_avg += grad_norm_pred_clipped_batch

                if epoch_idx % log_freq == 0:

                    pos_loss_avg /= self.config["steps_prediction"]
                    neg_loss_avg /= self.config["steps_prediction"]
                    loss_avg /= self.config["steps_prediction"]
                    grad_norm_pred_unclipped_avg /= self.config["steps_prediction"]
                    grad_norm_pred_clipped_avg /= self.config["steps_prediction"]

                    print('{} | {:.5f} | {:.5f} | {:.5f} | {:.3f} | {:.3f} | {:.2f}'.format( epoch_idx, loss_avg,pos_loss_avg,neg_loss_avg,
                          grad_norm_pred_unclipped_avg, grad_norm_pred_clipped_avg,
                          time.time() - start_time))

                    if best_loss > loss_avg:
                        self.saver.save(sess,
                                    os.path.join(checkpoint_dir,"model.ckpt"),
                                    global_step=epoch_idx // log_freq)
                        best_loss = copy.copy(loss_avg)
                    loss_avg = 0;pos_loss_avg=0;neg_loss_avg = 0;
                    grad_norm_pred_unclipped_avg, grad_norm_pred_clipped_avg = 0, 0
            self.saver.save(sess,
                        os.path.join(checkpoint_dir,"model.ckpt"),
                        global_step=epoch_idx // log_freq + 1)

    def initialize_model(self, sess):
            sess.run(tf.global_variables_initializer())
        
    def load(self,checkpoint_dir,sess):
        self.initialize_model(sess=sess)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          fname = os.path.join(checkpoint_dir, ckpt_name)
          self.saver.restore(sess, fname,)
          print(" [*] Load SUCCESS: %s" % fname)    
        else:
            print(" [*] Load Failed")
            
    def summarize(self,checkpoint_dir=None,test=False,B=40):
        if checkpoint_dir is None:
            try:
                checkpoint_dir = self.checkpoint_dir
            except Exception:
                checkpoint_dir = ''
        with tf.Session() as sess:
            self.load(checkpoint_dir,sess)    
            ds_length = len(self.ds.tst_data) if test else len(self.ds.trn_data)
            phi = []
            for d1 in range(ds_length)[:5]:
                phi_d = []
                for k in range(B):
                    X,I = [],[]
                    for i in range(self.config['batch_size']):
                        X_pair, X_pair_info,Y_pair = self.ds.sample_batch_pairs(positive=True,
                                                                                first_element=d1,
                                                                                test=test)
                        X.append(X_pair)
                        I.append(X_pair_info)
                    X,I = np.vstack(X),np.vstack(I)
                    phi_b = sess.run(self.model.phi,feed_dict={self.model.X: X,
                                        self.model.N_features: I[:,1],
                                        self.model.N_classes:  I[:,2],
                                        self.model.N_instances: I[:,0],})
                    phi_d.append(phi_b)
                phi_d = np.mean(np.vstack(phi_d),axis=0)
                phi.append(phi_d)
        return phi
