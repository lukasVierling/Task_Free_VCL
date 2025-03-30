# library imports
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset,Subset
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

import collections
# my imports
from utils.utils import kl_div_gaussian_layer,get_mle_estimate_extension, get_standard_normal_prior_extension


def evaluate_model(task_idx, model, test_datasets, batch_size,device ,num_samples,routing_mode, calculation_mode, var):
    # get a new distribution q
    accs = []
    heads_chosen_stat = []
    prev_head = model.get_active_head_idx()
    #optimize the KL between q_t tilde and q_t with likelihood over coreset_t
    for _, test_dataset in enumerate(test_datasets[:task_idx+1]):
        #coreset is a small dataset
        #activate the head for this dataset
        #test on the test dataset
        acc, heads_chosen = perform_predictions(model, test_dataset, batch_size, device, routing_mode=routing_mode)
        accs.append(acc)
        heads_chosen_stat.append(heads_chosen)
        print("test acc after additional finetuning: ",accs[-1])
        print("Heads chosen statistics:",heads_chosen_stat[-1])
        #reset the finetuning
    model.activate_head(prev_head)
    mean_accs = sum(accs)/len(accs)
    return mean_accs, heads_chosen_stat


def perform_predictions(model, curr_test_dataset,batch_size,device, num_samples=100, routing_mode="batchwise", calculation_mode="sampling", var=0.01):
    data_loader = DataLoader(curr_test_dataset, batch_size=batch_size)
    labels = []
    squared_errors = []
    correct = 0
    model.eval()
    accs = []
    heads_chosen = collections.defaultdict(int)
    with torch.no_grad():
        for x,y in tqdm(data_loader,desc="Testing Performance"):
            x,y = x.to(device),y.to(device)

            x = x.view(x.shape[0],-1)
            #evaluate the integral over weights via monte carlo estimate
            #preds, best_head = torch.ones((x.shape[0],model.output_dim)), 0
            preds, best_head = model.forward_with_routing(x, routing_mode=routing_mode)
            #update head statistics
            heads_chosen[best_head] += x.shape[0]

            pred = torch.argmax(preds, dim=-1)
            correct += torch.sum(pred == y)
            #print(pred.shape)
            labels.extend(pred.tolist())
            
    model.train()
    labels = torch.tensor(labels)
    #print("Tested with accuracy: " ,correct/len(labels))
    metrics = correct/len(labels)
    return metrics, heads_chosen

def vcl(model, train_datasets, test_datasets, batch_size, epochs, lr, device="cpu", baseline_window_size=50, current_window_size=1, c=5, num_samples=25, var=0.01, calculation_mode="sampling", routing_mode="batchwise",automatic_detection=False):
    '''
    Input:
    - prior : prior distribution
    - dataset : list of datasets
    Output: 
    - [(q_t,p_t)] t=1,...,T : Variational and predictive distribution at each step
    '''
    model.to(device)
    ret = collections.defaultdict(list)
    print(f"Detection uses baseline window size: {baseline_window_size} current window size: {current_window_size} and a std factor of {c}")
    
    #prior in the model is already implemented all values sampled from gaussian
    prior = None
    #init prior as N(0,1)
    #prior = get_standard_normal_prior_extension(model, device)
    #get MLE estimate
    #model_init = get_mle_estimate_extension(model, train_datasets[0], device)
    # get the number of datasets T
    #model.set_var_dist(model_init["encoder"])TODO later remove
    #model.set_heads(model_init["heads"])
    #print("Acc:", perform_predictions(model, test_datasets[0],256,device))

    #logs for plots:
    train_losses = []
    recon_losses = []
    accs = []
    tasks_to_heads_chosen = []

    T = len(train_datasets)
    for i in tqdm(range(T), desc="Training on tasks..."):
        #add task specific head to the model
        # get the current dataset D_i
        model.add_head()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        curr_dataset = train_datasets[i]

        print("permutation of current dataset is :", curr_dataset.get_permutation()[0])
        
        # Update the variational distribution

        data_loader = DataLoader(curr_dataset, batch_size=batch_size, shuffle=True) #TODO batch_size hyperparameter
        
        for epoch in tqdm(range(epochs), desc="Training"):
            pbar = tqdm(enumerate(data_loader), desc=f"Training in Epoch: {epoch}")
            for batch_idx,(x,y) in pbar: 
                x,y = x.to(device),y.to(device)
                x = x.view(x.shape[0],-1)
                optimizer.zero_grad()
                model.zero_grad()
                #forward through the model to get likelihood
                output, recon_loss = model(x,return_reconstruction_loss="nll") #-> returns [B,C]
                #calc the likelihood
                lhs = F.cross_entropy(output, y)
                # calculate the KL div between prior and new var dist -> closed form since both mena field gaussian
                rhs = model.kl_divergence()/len(curr_dataset)

                #rhs = 0
                #print(lhs)
                #print(rhs)
                loss = lhs + rhs + recon_loss # - because we want to maximize ELBO so minimize negative elbo

                loss.backward()
                optimizer.step()
                #log metrics
                train_losses.append(loss.cpu().item())
                recon_losses.append(recon_loss.cpu().item())

                pbar.set_description(f"Loss {loss.cpu().item()}, recon loss: {recon_loss.cpu().item()}")
                

        # Compute the final variational distribution (only used for prediction, and not propagation)
        #get q_t tilde before optimizing to get q_t
        acc, heads_chosen = evaluate_model(i, model, test_datasets, batch_size, device, num_samples=num_samples,routing_mode=routing_mode,calculation_mode=calculation_mode,var=var)

        tasks_to_heads_chosen.append(heads_chosen)
        # Perform prediction at test input x*
        #preds = perform_predictions(model, curr_test_dataset,device)
        #collect intermediate results
        print(f"Average RMSE of {acc.item()} after training on task {i}")
        accs.append(acc.item())
        #heads_chosen_list.append(heads_chosen)
        # init the model with the previous prior so we are not influenced by the coreset training
        #model.set_var_dist(prior)TODO reconsider the placement -> should not be used after last epocch
    #return final resutls
    #turn the dict into an object that we can store in the json file
    ret["heads_chosen"] = tasks_to_heads_chosen
    ret["acc"] = accs
    ret["train_losses"] = train_losses
    ret["recon_losses"] = recon_loss
    ret["heads_chosen"] = tasks_to_heads_chosen
    #print(ret)
    return ret


