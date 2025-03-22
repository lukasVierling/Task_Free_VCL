# library imports
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset,Subset
from tqdm import tqdm
import copy

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

def new_coreset(coreset_idx, curr_dataset, train_datasets, coreset_size=500, heuristic="random"):
    '''
    Input:
    - prev_coreset: the previous coreset C_{t-1}
    - curr_dataset: the current dataset D_t
    - heuristic: the heuristic that should be used to generate the new coreset
    Output:
    - new_coreset: the new coreset including samples from previous coreset and new dataset (is a list of datasets)
    '''
    new_coreset = None
    new_coreset_idx = None
    if coreset_size > 0:
        if heuristic == "random":
            #Select K random data points
            idx_list = torch.randperm(len(curr_dataset))[:coreset_size]
            #update the coreset idx list for the current datset
            new_coreset_idx = copy.deepcopy(coreset_idx)
            new_coreset_idx.append(idx_list)
            #iterate through the new list and create the current coreset
            coresets = []
            if len(coreset_idx) > 0:
                print("Concat old coresets")
                for task_no, coreset_idx_list in enumerate(coreset_idx):
                    coresets.append(Subset(train_datasets[task_no], coreset_idx_list))
            coresets.append(Subset(curr_dataset, idx_list))
            new_coreset = coresets #IS a list
        else:
            print("Other heuristics are not implemented, please set heuristic to one of those values [random,]")
    else: 
        new_coreset = None #not using any coresets 
        print("Not using any coresets.")

    return new_coreset, new_coreset_idx

def update_var_approx_non_coreset(model, prior, curr_dataset, coreset_idx, train_datasets, batch_size, epochs, lr, device):
    # get a new distribution q
    #optimize the parameters of the new var_approx
    if coreset_idx is not None:
        #get coreset indices of newest datset
        idx_list = coreset_idx[-1]
        #invert the coreset
        inverted_idx_list = [i for i in range(len(curr_dataset))if i not in idx_list]
        #check if everything is alright
        print(f"length of inverted coreset: {len(inverted_idx_list)} and length of coreset: {len(idx_list)} should be {len(curr_dataset)}")
        idx_list = inverted_idx_list
        train_dataset = Subset(curr_dataset, idx_list)
        print("Length of train dataset: ",len(train_dataset))
    else:
        train_dataset = curr_dataset
    #minimize the KL div
    minimize_KL(model, prior, train_dataset, batch_size, epochs, lr, device)

def minimize_KL(model, prior, dataset, batch_size, epochs, lr, device):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #TODO batch_size hyperparameter
    for epoch in tqdm(range(epochs), desc="Training"):
        for x,y in tqdm(data_loader, desc=f"Training in Epoch: {epoch}"): 
            x,y = x.to(device),y.to(device)
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

def update_final_var_dist_and_test(task_idx, model, prior ,prior_heads, curr_coreset, test_datasets, batch_size, epochs, lr, device):
    # get a new distribution q
    accs = []
    prev_head = model.get_active_head_idx()
    #optimize the KL between q_t tilde and q_t with likelihood over coreset_t
    if curr_coreset is not None:
        for idx, coreset in enumerate(curr_coreset[:task_idx+1]):
            #coreset is a small dataset
            #activate the head for this dataset
            model.activate_head(idx)
            #eval acc
            print("test acc before additional finetuning: ",perform_predictions(model, test_datasets[idx], batch_size, device))
            #finetune on the coreset
            minimize_KL(model, prior, coreset ,batch_size, epochs, lr, device)
    
    #start evaluating the model on all test sets
    for idx, test_dataset in enumerate(test_datasets[:task_idx+1]):
        #coreset is a small dataset
        #activate the head for this dataset
        model.activate_head(idx)
        #test on the test dataset
        accs.append(perform_predictions(model, test_dataset,batch_size, device))
        print("test acc after additional finetuning: ", accs[-1])
        #reset the finetuning
    
    #reset the finetuning
    model.set_var_dist(prior)
    model.set_heads(prior_heads)
    model.activate_head(prev_head)
    mean_accs = sum(accs)/len(accs)
    return mean_accs



def perform_predictions(model, curr_test_dataset,batch_size,device):
    data_loader = DataLoader(curr_test_dataset, batch_size=batch_size)
    labels = []
    correct = 0
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(data_loader,desc="Testing Performance"):
            x,y = x.to(device),y.to(device)
            probs = model(x)
            pred = torch.argmax(probs, dim=-1)
            correct += torch.sum(pred == y)
            #print(pred.shape)
            labels.extend(pred.tolist())

    model.train()
    labels = torch.tensor(labels)
    #print("Tested with accuracy: " ,correct/len(labels))
    acc = correct/len(labels)
    return acc

def coreset_vcl(model, train_datasets, test_datasets, batch_size, epochs, lr, coreset_size=0, coreset_heuristic="random", device="cpu"):
    '''
    Input:
    - prior : prior distribution
    - dataset : list of datasets
    Output: 
    - [(q_t,p_t)] t=1,...,T : Variational and predictive distribution at each step
    '''
    model.to(device)
    ret = []
    use_coreset = coreset_size > 0
    coreset_idx = []
    if not(use_coreset):
        print("Not using a coreset")
    
    else:
        print("Initialize an empty coreset")
    #prior in the model is already implemented all values sampled from gaussian
    prior = model.get_var_dist()
    # get the number of datasets T
    T = len(train_datasets)
    for i in tqdm(range(T), desc="Training on tasks..."):
        #add task specific head to the model
        model.add_head()
        #TODO do we have to reinitialize the model here? 
        # get the current dataset D_i
        curr_dataset = train_datasets[i]
        curr_test_dataset = test_datasets[i]
        # update the coreset with D_i
        curr_coresets, coreset_idx = new_coreset(coreset_idx, curr_dataset, train_datasets, coreset_size=coreset_size, heuristic=coreset_heuristic)
        if use_coreset:
            print(f"current Coreset length (should be {coreset_size * (i+1)}): ", len(curr_coresets))
        # Update the variational distribution for non-coreset data points
        update_var_approx_non_coreset(model, prior, curr_dataset, coreset_idx, train_datasets ,batch_size, epochs, lr, device)
        # Compute the final variational distribution (only used for prediction, and not propagation)
        #get q_t tilde before optimizing to get q_t
        prior = model.get_var_dist()
        heads = model.get_heads()
        #TODO think about if this is correct but should be because we only finetune the head here
        acc = update_final_var_dist_and_test(i, model, prior, heads, curr_coresets, test_datasets, batch_size, epochs, lr, device)
        # Perform prediction at test input x*
        #preds = perform_predictions(model, curr_test_dataset,device)
        #collect intermediate results
        ret.append(acc)
        print(f"Average acc of {acc} after training on task {i}")
        # init the model with the previous prior so we are not influenced by the coreset training
        #model.set_var_dist(prior)TODO reconsider the placement -> should not be used after last epocch
    #return final resutls
    return ret




