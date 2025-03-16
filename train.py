# library imports
import torch
import numpy as np
from torch.utils.data import ConcatDataset

# my imports
from models.var_distribution import GaussianMeanField #TODO use this instead of always using model

def init_coreset_variational_approx(prior):
    '''
    Input:
    - prior: prior distirbution
    Output:
    - coreset: the initial coreset
    - var_approx: the initial variational approximation
    '''
    var_approx = prior
    return None, var_approx

def new_coreset(prev_coreset, curr_dataset, coreset_size=500, heuristic="random"):
    '''
    Input:
    - prev_coreset: the previous coreset C_{t-1}
    - curr_dataset: the current dataset D_t
    - heuristic: the heuristic that should be used to generate the new coreset
    Output:
    - new_coreset: the new coreset including samples from previous coreset and new dataset
    '''
    if heuristic == "random":
        #Select K random data points
        idx = torch.randperm(curr_dataset["X"].size(0))[:coreset_size]
        chosen_subset = curr_dataset[idx]
        new_coreset = ConcatDataset([prev_coreset, chosen_subset])
    else:
        print("Other heuristics are not implemented, please set heuristic to one of those values [random,]")

    return new_coreset

def update_var_approx_non_coreset(var_approx, model, dataset, curr_coreset, prev_coreset):
    pass

def update_final_var_dist(var_approx, model, curr_coreset):
    pass

def perform_predictions(final_var_dist, model, curr_test_dataset):
    pass

def coreset_vcl(prior, model, train_datasets, test_datasets):
    '''
    Input:
    - prior : prior distribution
    - dataset : list of datasets
    Output: 
    - [(q_t,p_t)] t=1,...,T : Variational and predictive distribution at each step
    '''
    ret = []
    #Init the coreset and q_0
    prev_coreset, var_approx = init_coreset_variational_approx(prior)
    # get the number of datasets T
    T = len(train_datasets)
    for i in range(T):
        # get the current dataset D_i
        curr_dataset = train_datasets[i]
        curr_test_dataset = test_datasets[i]
        # update the coreset with D_i
        curr_coreset = new_coreset(prev_coreset, curr_dataset)
        # Update the variational distribution for non-coreset data points
        var_approx = update_var_approx_non_coreset(var_approx, model, curr_dataset, curr_coreset, prev_coreset)
        # Compute the final variational distribution (only used for prediction, and not propagation)
        final_var_dist = update_final_var_dist(var_approx, model, curr_coreset)
        # Perform prediction at test input x*
        preds = perform_predictions(final_var_dist, model, curr_test_dataset)
        #collect intermediate results
        ret.append((final_var_dist, preds))
        prev_coreset = curr_coreset
    #return final resutls
    return ret




