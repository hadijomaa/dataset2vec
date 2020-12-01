#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:57:46 2020

@author: hsjomaa
"""

import torch
import numpy as np
import argparse
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList
from botorch.sampling.samplers import SobolQMCNormalSampler
import json
import pandas as pd
import os
import sys
from scipy.stats import norm
from helper_fn import regret,warm_start
torch.manual_seed(29)
np.random.seed(327)

# suppress GPyTorch warnings about adding jitter
import warnings
warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--posterior', help='number of posterior samples', type=int,default=64)
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--k', help='number of starting points', type=int,default=20)
parser.add_argument('--index', help='Dataset index in the fold', type=int,default=0)
parser.add_argument('--method', help='Metafeatures used', choices=['aaai','tstr','d2v','random','transferable'],default='d2v')
parser.add_argument('--searchspace', type=str,default='a')

args = parser.parse_args()

device = torch.device("cpu")
dtype = torch.double
currentdir     = os.path.dirname(os.path.realpath(__file__))
rootdir   = '/'.join(currentdir.split('/')[:-1])
sys.path.insert(0,rootdir)
from dataset import Dataset

savedir     = os.path.join(rootdir,"results",f"rgpe-samples-{args.posterior}",f"searchspace-{args.searchspace}",f"init-{args.k}",args.method,f"split-{args.split}" if args.method=="d2v" else "")
os.makedirs(savedir,exist_ok=True)


splits_file       = os.path.join(rootdir, "metadataset",f"searchspace-{args.searchspace}",f"searchspace-{args.searchspace}-splits.csv")
metasplits        = pd.read_csv(splits_file,index_col=0)
info_file       = os.path.join(rootdir, "metadataset"  ,"info.json")
configuration   = json.load(open(info_file,'r'))[args.searchspace]

# specific to rgpe
configuration['split']                     = args.split
configuration['searchspace']               = args.searchspace
configuration['nrandom']                   = args.k
configuration['noise_std']                 = 0.01
configuration['minmax']                    = True
configuration['num_training_points']       = configuration["cardinality"]
configuration['NUM_POSTERIOR_SAMPLES']     = args.posterior

normalized_dataset         = Dataset(configuration,rootdir,use_valid=True)

def propose(optimal, gprm,indices):
    outputs       = gprm.forward(torch.tensor(normalized_dataset.orighyper))
    mu, sigma     = outputs.mean,outputs.stddev
    mu            = mu.detach().numpy().reshape(-1,)
    sigma         = sigma.detach().numpy().reshape(-1,)
    with np.errstate(divide='warn'):
        imp = mu - optimal
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    for _ in range(len(ei)):
        if _ in indices:
            ei[_] = 0
    return np.argmax(ei)

def get_fitted_model(train_X, train_Y, train_Yvar, state_dict=None):
    """
    Get a single task GP. The model will be fit unless a state_dict with model 
        hyperparameters is provided.
    """
    Y_mean = train_Y.mean(dim=-2, keepdim=True)
    Y_std = train_Y.std(dim=-2, keepdim=True)
    model = FixedNoiseGP(train_X, (train_Y - Y_mean)/Y_std, train_Yvar)
    model.Y_mean = Y_mean
    model.Y_std = Y_std
    if state_dict is None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
    else:
        model.load_state_dict(state_dict)
    return model

def roll_col(X, shift):  
    """
    Rotate columns to right by shift.
    """
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)


def compute_ranking_loss(f_samps, target_y):
    """
    Compute ranking loss for each sample from the posterior over target points.
    
    Args:
        f_samps: `n_samples x (n) x n`-dim tensor of samples
        target_y: `n x 1`-dim tensor of targets
    Returns:
        Tensor: `n_samples`-dim tensor containing the ranking loss across each sample
    """
    n = target_y.shape[0]
    if f_samps.ndim == 3:
        # Compute ranking loss for target model
        # take cartesian product of target_y
        cartesian_y = torch.cartesian_prod(
            target_y.squeeze(-1), 
            target_y.squeeze(-1),
        ).view(n, n, 2)
        # the diagonal of f_samps are the out-of-sample predictions
        # for each LOO model, compare the out of sample predictions to each in-sample prediction
        rank_loss = ((f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps) ^ (cartesian_y[..., 0] < cartesian_y[..., 1])).sum(dim=-1).sum(dim=-1)
    else:
        rank_loss = torch.zeros(f_samps.shape[0], dtype=torch.long, device=target_y.device)
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        for i in range(1,target_y.shape[0]):
            rank_loss += ((roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack)).sum(dim=-1) 
    return rank_loss


def get_target_model_loocv_sample_preds(train_x, train_y, train_yvar, target_model, num_samples):
    """
    Create a batch-mode LOOCV GP and draw a joint sample across all points from the target task.
    
    Args:
        train_x: `n x d` tensor of training points
        train_y: `n x 1` tensor of training targets
        target_model: fitted target model
        num_samples: number of mc samples to draw
    
    Return: `num_samples x n x n`-dim tensor of samples, where dim=1 represents the `n` LOO models,
        and dim=2 represents the `n` training points.
    """
    batch_size = len(train_x)
    masks = torch.eye(len(train_x), dtype=torch.uint8, device=device).bool()
    train_x_cv = torch.stack([train_x[~m] for m in masks])
    train_y_cv = torch.stack([train_y[~m] for m in masks])
    train_yvar_cv = torch.stack([train_yvar[~m] for m in masks])
    state_dict = target_model.state_dict()
    # expand to batch size of batch_mode LOOCV model
    state_dict_expanded = {name: t.expand(batch_size, *[-1 for _ in range(t.ndim)]) for name, t in state_dict.items()}
    model = get_fitted_model(train_x_cv, train_y_cv, train_yvar_cv, state_dict=state_dict_expanded)
    with torch.no_grad():
        posterior = model.posterior(train_x)
        # Since we have a batch mode gp and model.posterior always returns an output dimension,
        # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
        # the last dimension.
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        return sampler(posterior).squeeze(-1)
    

def compute_rank_weights(train_x,train_y, base_models, target_model, num_samples):
    """
    Compute ranking weights for each base model and the target model (using 
        LOOCV for the target model). Note: This implementation does not currently 
        address weight dilution, since we only have a small number of base models.
    
    Args:
        train_x: `n x d` tensor of training points (for target task)
        train_y: `n` tensor of training targets (for target task)
        base_models: list of base models
        target_model: target model
        num_samples: number of mc samples
    
    Returns:
        Tensor: `n_t`-dim tensor with the ranking weight for each model
    """
    ranking_losses = []
    # compute ranking loss for each base model
    for task in range(len(base_models)):
        model = base_models[task]
        # compute posterior over training points for target task
        posterior = model.posterior(train_x)
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        # compute and save ranking loss
        ranking_losses.append(compute_ranking_loss(base_f_samps, train_y))
    # compute ranking loss for target model using LOOCV
    # f_samps
    target_f_samps = get_target_model_loocv_sample_preds(train_x, train_y, train_yvar, target_model, num_samples)
    ranking_losses.append(compute_ranking_loss(target_f_samps, train_y))
    ranking_loss_tensor = torch.stack(ranking_losses)
    # compute best model (minimum ranking loss) for each sample
    best_models = torch.argmin(ranking_loss_tensor, dim=0)
    # compute proportion of samples for which each model is best
    rank_weights = best_models.bincount(minlength=len(ranking_losses)).type_as(train_x)/num_samples
    return rank_weights

def f(X,file):
    """
    Torch-compatible objective function for the target_task
    """
    if len(X.shape)==1:
        X = X[None]
    idx = [np.where((_==normalized_dataset.orighyper).all(axis=1))[0][0] for _ in X]
    outputs = np.vstack([normalized_dataset.global_surr[file][:,0][_] for _ in idx])
    return torch.tensor(outputs,dtype=dtype)

class RGPE(GP, GPyTorchModel):
    """
    Rank-weighted GP ensemble. Note: this class inherits from GPyTorchModel which provides an 
        interface for GPyTorch models in botorch.
    """
    def __init__(self, models, weights):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self._num_outputs = 1
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        self.to(dtype=weights.dtype, device=weights.device)
        
    def forward(self, x):
        weighted_means = []
        weighted_covars = []
        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights**2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()
        
        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            model = self.models[raw_idx]
            posterior = model.posterior(x)
            # unstandardize predictions
            posterior_mean = posterior.mean.squeeze(-1)*model.Y_std + model.Y_mean
            posterior_cov = posterior.mvn.lazy_covariance_matrix * model.Y_std.pow(2)
            # apply weight
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            weighted_covars.append(posterior_cov * weight**2)
        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)

metafeatures = {'aaai':pd.read_csv(os.path.join(rootdir,"metafeatures","mf1.csv"),index_col=0,header=0),
                    'tstr':pd.read_csv(os.path.join(rootdir,"metafeatures","mf2.csv"),index_col=0,header=0),
                    'd2v':pd.read_csv(os.path.join(rootdir,"metafeatures",f"meta-features-split-{args.split}.csv"),index_col=0,header=0)}
    
files = metasplits[f"train-{args.split}"].dropna().tolist()+metasplits[f"valid-{args.split}"].dropna().tolist()+metasplits[f"test-{args.split}"].dropna().tolist()

file = files[args.index]

base_tasks = [_ for _ in files if _ != file]
assert(len(base_tasks)==119)
NUM_BASE_TASKS  = len(base_tasks)
data_by_task    = {}
base_model_list = []

for task in range(NUM_BASE_TASKS):
    raw_x = normalized_dataset.orighyper
    f_x         = f(raw_x,file=base_tasks[task])
    train_y     = f_x + configuration['noise_std']*torch.randn_like(f_x)
    train_yvar  = torch.full_like(train_y, configuration['noise_std']**2)
    # store training data
    data_by_task[task] = {
        'train_x': torch.tensor(raw_x),
        'train_y': train_y,
        'train_yvar': train_yvar,
        }
    model = get_fitted_model(data_by_task[task]['train_x'], 
                              data_by_task[task]['train_y'], 
                              data_by_task[task]['train_yvar'],
                              )
    base_model_list.append(model)  
    
    
# Initial random observations
indices     = warm_start(method=args.method,dataset=normalized_dataset,k=args.k,file=file,metafeatures=metafeatures)
train_x     = np.vstack([normalized_dataset.orighyper[_] for _ in indices])
train_y     = f(train_x,file=file) 
train_yvar  = torch.full_like(train_y, configuration['noise_std']**2)
# keep track of the best observed point at each iteration
best_value  = train_y.max().item()
train_x     = torch.tensor(train_x)
opt         = max(normalized_dataset.global_surr[file][:,0])
indices = list(indices)
# Run N_BATCH rounds of BayesOpt after the initial random batch
for iteration in range(args.n_iters): 
    incumbents = [normalized_dataset.global_surr[file][_,0] for _ in indices]
    if opt in incumbents:
        break
    target_model = get_fitted_model(train_x, train_y, train_yvar)
    model_list   = base_model_list + [target_model]
    rank_weights = compute_rank_weights(train_x, train_y, base_model_list, target_model, configuration['NUM_POSTERIOR_SAMPLES'], )
    # create model and acquisition function
    rgpe_model = RGPE(model_list, rank_weights)
    candidate = propose(best_value,rgpe_model,indices)
    indices  +=[candidate]
    new_x     = normalized_dataset.orighyper[candidate]
    new_y     = f(new_x,file=file)
    new_yvar  = torch.full_like(new_y, configuration['noise_std']**2)

    # update training points
    train_x     = torch.cat((train_x, torch.tensor(new_x[None])))
    train_y     =  torch.cat((train_y, new_y))
    train_yvar  = torch.cat((train_yvar, new_yvar))

    # get the new best observed value
    best_value = train_y.max().item()
    
response = normalized_dataset.global_surr[file][:,0]
output   = [response[_] for _ in indices]
output   = np.hstack(output).reshape(-1,)

results            = regret(output,response)
results['indices'] = np.asarray(indices).reshape(-1,)

results.to_csv(os.path.join(savedir,f"{file}.csv"))