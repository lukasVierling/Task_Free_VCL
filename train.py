# library imports
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

# my imports
from utils.utils import kl_div_gaussian

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
    new_coreset = None
    if curr_dataset:
        if coreset_size > 0:
            if heuristic == "random":
                #Select K random data points
                idx = torch.randperm(len(curr_dataset))[:coreset_size]
                # select subset of the datset based on the idx list
                chosen_subset = torch.utils.data.Subset(curr_dataset, idx.tolist())
                # concat the prev_coreset with chosen_subset
                new_coreset = ConcatDataset([prev_coreset, chosen_subset])
            else:
                print("Other heuristics are not implemented, please set heuristic to one of those values [random,]")
        else: 
            new_coreset = None
    else:
        print("Please provide a non empty dataset")
    return new_coreset

def update_var_approx_non_coreset(model, prior, dataset, curr_coreset, prev_coreset):
    # get a new distribution q
    #optimize the parameters of the new var_approx
    if curr_coreset and prev_coreset:
        '''
         merged_ds = ConcatDataset([dataset, prev_coreset])

        #TODO filter the merged_ds for elements in curr_coreset
        #filter elements from curr_coreset out of merged_ds
        filtered_list = []
        merged_ds_loader = DataLoader(merged_ds)
        for x,y in merged_ds_loader:
            if x not in curr_coreset or y not in curr_coreset:
                filtered_list.apend( [x,y] )
        filtered_ds = TensorDataset(filtered_list)
        '''
        merged_ds = None
        print("No coreset support implemented yet.")
    else:
        merged_ds = dataset
    #minimize the KL div
    minimize_KL(model, prior, merged_ds)

def minimize_KL(model, prior, dataset):
    
    optimizer = torch.optim.Adam(model.parameters())

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True) #TODO batch_size hyperparameter

    for x,y in data_loader: 
        optimizer.zero_grad()
        #forward through the model to get likelihood
        output = model(x) #-> returns [B,C]
        #calc the likelihood
        probs = output.gather(1, y.view(-1, 1)).squeeze() # index output to get [B]
        log_probs = torch.log(probs + 1e-8) #for numeric stability
        lhs = log_probs.sum()
        # calculate the KL div between prior and new var dist -> closed form since both mena field gaussian
        rhs = kl_div_gaussian(model.get_var_dist(), prior)
        loss = -(lhs - rhs) # - because we want to maximize ELBO so minimize negative elbo
        loss.backward()
        optimizer.step()

def update_final_var_dist(model, prior, curr_coreset):
    # get a new distribution q

    #optimize the KL between q_t tilde and q_t with likelihood over coreset_t
    minimize_KL(model, prior, curr_coreset)


def perform_predictions(model, curr_test_dataset):
    data_loader = DataLoader(curr_test_dataset)
    labels = []
    correct = 0
    model.eval()
    with torch.no_grad():
        for x,y in data_loader:
            probs = model(x)
            pred = torch.argmax(probs, dim=-1)
            correct += torch.sum(pred == y)
            labels.append(pred)

    model.train()
    labels = torch.tensor(labels)
    print("Tested with accuracy: " ,correct/len(labels))
    return labels

def coreset_vcl(model, train_datasets, test_datasets, coreset_size=0, coreset_heuristic="random"):
    '''
    Input:
    - prior : prior distribution
    - dataset : list of datasets
    Output: 
    - [(q_t,p_t)] t=1,...,T : Variational and predictive distribution at each step
    '''
    ret = []
    if coreset_size == 0:
        print("Not using a coreset")
        prev_coreset = None
    else:
        print("Initialize an empty coreset:")
        # Initialize prev_coreset based on train_datasets[0]
        first_dataset = train_datasets[0]  # Assuming train_datasets is a list of TensorDataset objects
        # get the shape of X and dtype of Y 
        img_shape = first_dataset.tensors[0].shape[1:]
        label_dtype = first_dataset.tensors[1].dtype 
        # now intiialize the prev_coreset as empty for the first iteration
        prev_coreset = TensorDataset(
            torch.empty((0, *img_shape)),  # empty image
            torch.empty((0,), dtype=label_dtype) #empty label
        )
    print("Start with coreset:", prev_coreset)
    #prior in the model is already implemented all values sampled from gaussian
    prior = model.get_var_dist()
    # get the number of datasets T
    T = len(train_datasets)
    for i in range(T):
        #add task specific head to the model
        model.add_head()
        # get the current dataset D_i
        curr_dataset = train_datasets[i]
        curr_test_dataset = test_datasets[i]
        # update the coreset with D_i
        curr_coreset = new_coreset(prev_coreset, curr_dataset, coreset_size=coreset_size, heuristic=coreset_heuristic)
        print("current Coreset: ", curr_coreset)
        # Update the variational distribution for non-coreset data points
        update_var_approx_non_coreset(model, prior, curr_dataset, curr_coreset, prev_coreset)
        # Compute the final variational distribution (only used for prediction, and not propagation)
        #get q_t tilde before optimizing to get q_t
        prior = model.get_var_dist()
        if coreset_size > 0:
            print("Perform anohter round of training because we are using a coreset.")
            update_final_var_dist(model, prior, curr_coreset)
        # Perform prediction at test input x*
        preds = perform_predictions(model, curr_test_dataset)
        #collect intermediate results
        var_dist = {
            "shared": model.get_var_dist(),
            "head_idx": i
        }
        ret.append((var_dist, preds))
        prev_coreset = curr_coreset
    #return final resutls
    return ret




