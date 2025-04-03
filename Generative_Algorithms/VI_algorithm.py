# library imports
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset,Subset
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import math
import os
import torch.nn.functional as F
# my imports
from utils.utils import kl_div_gaussian_for_gen, get_standard_normal_prior_gen

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

def vae_loss(recon_x, x, mean, log_var):
    #eps = 1e-6 #for stability
    #likelihood part
    bernoulli_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #now the KL part
    kl = - 0.5 * torch.sum(1+log_var - mean**2 - torch.exp(log_var))
    return bernoulli_loss + kl #both terms are positive but then flip sign later

def minimize_KL(model, prior, dataset, batch_size, epochs, lr, device):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #TODO batch_size hyperparameter
    if not(prior):
        print("Prior is None")
    for epoch in tqdm(range(epochs), desc="Training"):
        pbar = tqdm(data_loader, desc=f"Training in Epoch: {epoch}")
        for x,y in pbar: 
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            model.zero_grad()
            #forward through the model to get likelihood
            output, mean, log_var = model(x) #-> returns [B,C]
            #calc the VAE loss
            
            lhs = vae_loss(output, x, mean, log_var) #this is mean!
            
            # calculate the KL div between prior and new var dist -> closed form since both mena field gaussian
            if prior is not None:
                rhs = kl_div_gaussian_for_gen(model.get_var_dist(detach=False), prior) #TODO this has previously not been correct
                rhs = batch_size * rhs / len(dataset) #TODO divide by len dataset
                #rhs = 0
            else:
                rhs = 0
            loss = lhs + rhs # - because we want to maximize ELBO so minimize negative elbo Note: no sign flip for lhs because already nll
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Current lhs: {lhs} and Current rhs: {rhs}")

def update_final_var_dist_and_test(task_idx, model, prior ,prior_heads,prior_encoders, classifier, curr_coreset, test_datasets, batch_size, epochs, lr, device):
    # get a new distribution q
    uncertainties = []
    llhs = []
    prev_head = model.get_active_head_idx()
    prev_encoder = model.get_active_encoder_idx()

    #optimize the KL between q_t tilde and q_t with likelihood over coreset_t
    if curr_coreset is not None:
        for idx, coreset in enumerate(curr_coreset[:task_idx+1]):
            #print("Length of core sets:" len(curr_coreset[:task_idx+1]))
            #print("test dataset idx:", idx)
            #coreset is a small dataset
            #activate the head for this dataset
            model.activate_head(idx)
            model.activate_encoder(idx)
            print("test acc before additional finetuning: ",perform_generations(model, classifier, test_datasets[idx], batch_size, device))
            minimize_KL(model, prior, coreset ,batch_size, epochs, lr, device)
            #obtained finetuned model for dataset head_idx
            #test on the test dataset
            uncertainty, llh = perform_generations(model, classifier, test_datasets[idx], batch_size, device)
            uncertainties.append(uncertainty)
            llhs.append(llh)
            print("test acc after additional finetuning: ",uncertainties[-1], llhs[-1])
            #reset the finetuning
            model.set_var_dist(prior)
            model.set_heads(prior_heads)
            model.set_encoders(prior_encoders)
    else:
        for idx, test_dataset in enumerate(test_datasets[:task_idx+1]):
            #coreset is a small dataset
            #activate the head for this dataset
            model.activate_head(idx)
            model.activate_encoder(idx)
            #test on the test dataset
            uncertainty, llh = perform_generations(model, classifier, test_datasets[idx], batch_size, device)
            uncertainties.append(uncertainty)
            llhs.append(llh)
            print("test acc after additional finetuning: ",uncertainties[-1], llhs[-1])
            #reset the finetuning
    model.activate_head(prev_head)
    model.activate_encoder(prev_encoder)

    return uncertainties, llhs

def perform_generations(model, classifier, curr_test_dataset,batch_size,device, num_samples=1000):
    model.eval()
    label = curr_test_dataset.label
    classifier.to(device)
    uncertainty_measure = 0
    ###
    #uncertainty part
    ###
    with torch.no_grad():
        #get the noise vectors
        z = torch.randn(num_samples, model.latent_dim).to(device)
        # decode the vectors with the model
        #calc the integral over weights
        flat_images = 0
        for _ in range(num_samples):
            flat_images += model.decode(z).view(num_samples, 1, 28, 28)
        flat_images = flat_images/num_samples
        #classify the images
        probs = classifier.get_probs(flat_images)
        #calc the KL div with the one-hot vector
        one_hot = F.one_hot(torch.full((num_samples,), label, device=device), num_classes=10).float() #hardcode 10 for mnist
        kl_div = F.kl_div(torch.log(probs), one_hot, reduction="mean") #expects log probabilities
        uncertainty_measure = kl_div


        ###
        #llh part
        ###
    
        #go over test set
        summed_ll = 0
        test_loader = DataLoader(curr_test_dataset, shuffle=False, batch_size = 1)
        K =100
        for x,_ in test_loader:
            x = x.to(device)
            batch_size = x.shape[0]
            #print("Batch Size: ", batch_size)
            #print("K:", K)

            x_rep = x.unsqueeze(0).repeat(K,1,1)
            x_rep = x_rep.view(batch_size * K, -1)

            #obtain mean and var
            mean, log_var = model.encode(x_rep)
            std = torch.exp(log_var * 0.5)
            var = torch.exp(log_var)

            #obtain z
            eps = torch.randn_like(std)
            z = mean + std*eps

            #obtain log probs
            log_q_z = -0.5*torch.log(2*math.pi*var) - (z-mean)**2 / (2* var)
            log_q_z = torch.sum(log_q_z, dim=1)
            log_p_z = -0.5* math.log(2*math.pi) - z**2 * 0.5
            log_p_z = torch.sum(log_p_z, dim=1)
            
            #obtain log likelihood
            img_mean = model.decode(z)
            eps = 1e-7 
            img_mean = img_mean.clamp(eps, 1 - eps)
            log_p_x_z = x * torch.log(img_mean) + (1-x) * torch.log(1-img_mean)
            log_p_x_z = log_p_x_z.view(batch_size*K, -1).sum(dim=1)

            log_p_x = log_p_x_z + log_p_z - log_q_z

            log_p_x = log_p_x.view(K, batch_size)

            #find maximum element for every "batch" of K elements
            max_log_w, _ = torch.max(log_p_x, dim=0, keepdim=True)
            shifted = log_p_x - max_log_w
            results = max_log_w + torch.log(torch.mean(torch.exp(shifted), dim=0))

            summed_ll += results.sum()

        average_ll = summed_ll / len(curr_test_dataset)
    model.train()
    return uncertainty_measure, average_ll

def coreset_vcl(model, train_datasets, test_datasets, classifier, batch_size, epochs, lr, coreset_size=0, coreset_heuristic="random", device="cpu"):
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
    prior = get_standard_normal_prior_gen(model,device) #inshallah klappt
    # get the number of datasets T
    T = len(train_datasets)
    for i in tqdm(range(T), desc="Training on tasks..."):
        #add task specific head to the model
        model.add_head()
        # add encoder
        model.add_encoder()
        #TODO do we have to reinitialize the model here? 
        # get the current dataset D_i
        curr_dataset = train_datasets[i]
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
        encoders = model.get_encoders()
        #perform_generations(model, curr_test_dataset, batch_size, device)
        #TODO think about if this is correct but should be because we only finetune the head here
        acc = update_final_var_dist_and_test(i, model, prior, heads, encoders, classifier, curr_coresets, test_datasets, batch_size, epochs, lr, device)
        # Perform prediction at test input x*
        #preds = perform_generations(model, curr_test_dataset,device)
        #collect intermediate results
        ret.append(acc)
        print(f"Average acc of {acc} after training on task {i}")
        # init the model with the previous prior so we are not influenced by the coreset training
        #model.set_var_dist(prior)TODO reconsider the placement -> should not be used after last epocch
    #return final resutls
    return ret




