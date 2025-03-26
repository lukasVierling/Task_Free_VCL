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


def evaluate_model(task_idx, model, test_datasets, batch_size,device ,num_samples):
    # get a new distribution q
    accs = []
    heads_chosen_stat = collections.defaultdict(int)
    prev_head = model.get_active_head_idx()
    #optimize the KL between q_t tilde and q_t with likelihood over coreset_t
    for _, test_dataset in enumerate(test_datasets[:task_idx+1]):
        #coreset is a small dataset
        #activate the head for this dataset
        #test on the test dataset
        acc, heads_chosen = perform_predictions(model, test_dataset, batch_size, device, num_samples)
        accs.append(acc)
        for head, times in heads_chosen.items():
            heads_chosen_stat[head] += times
        print("test acc after additional finetuning: ",accs[-1])
        print("Heads chosen statistics:",heads_chosen_stat)
        #reset the finetuning
    model.activate_head(prev_head)
    mean_accs = sum(accs)/len(accs)
    return mean_accs, heads_chosen_stat



def perform_predictions(model, curr_test_dataset,batch_size,device, num_samples=100):
    data_loader = DataLoader(curr_test_dataset, batch_size=batch_size)
    labels = []
    squared_errors = []
    correct = 0
    model.eval()
    heads_chosen = collections.defaultdict(int)
    with torch.no_grad():
        for x,y in tqdm(data_loader,desc="Testing Performance"):
            x,y = x.to(device),y.to(device)

            #evaluate the integral over weights via monte carlo estimate
            probs = []
            preds, best_head = model.forward_with_routing(x)
            #update head statistics
            heads_chosen[best_head] += x.shape[0]
            probs.append(preds)
            probs = torch.stack(probs).mean(dim=0)

            if model.mode == "regression":
                one_hot_labels = F.one_hot(y, num_classes=model.output_dim).float()
                se = F.mse_loss(probs, one_hot_labels, reduction="sum")
                squared_errors.append(se.item())
            else:
                pred = torch.argmax(probs, dim=-1)
                correct += torch.sum(pred == y)
                #print(pred.shape)
                labels.extend(pred.tolist())

    model.train()
    labels = torch.tensor(labels)
    #print("Tested with accuracy: " ,correct/len(labels))
    if model.mode == "regression":
        total = torch.tensor(sum(squared_errors))
        metrics = torch.sqrt(total /(len(curr_test_dataset) * model.output_dim))
    else:
        metrics = correct/len(labels)
    return metrics, heads_chosen

def vcl(model, train_datasets, test_datasets, batch_size, epochs, lr, device="cpu", baseline_window_size=50, current_window_size=1, c=5, num_samples=25):
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
    prior = get_standard_normal_prior_extension(model, device)
    #get MLE estimate
    model_init = get_mle_estimate_extension(model, train_datasets[0], device)
    # get the number of datasets T
    model.add_head()
    model.set_var_dist(model_init["encoder"])
    model.set_heads(model_init["heads"])
    print("Acc:", perform_predictions(model, test_datasets[0],256,device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #Mi setup
    baseline_mi = collections.deque(maxlen=baseline_window_size)
    current_mi = collections.deque(maxlen=current_window_size)

    #logs for plots:
    train_losses = []
    mutual_infos = []
    means = []
    stds = []

    T = len(train_datasets)
    for i in tqdm(range(T), desc="Training on tasks..."):
        #add task specific head to the model
        # get the current dataset D_i
        curr_dataset = train_datasets[i]

        print("permutation of current dataset is :", curr_dataset.get_permutation()[0])
        
        # Update the variational distribution

        data_loader = DataLoader(curr_dataset, batch_size=batch_size, shuffle=True) #TODO batch_size hyperparameter

        for epoch in tqdm(range(epochs), desc="Training"):
            for batch_idx,(x,y) in tqdm(enumerate(data_loader), desc=f"Training in Epoch: {epoch}"): 
                x,y = x.to(device),y.to(device)
                optimizer.zero_grad()
                model.zero_grad()
                #forward through the model to get likelihood
                output = model(x) #-> returns [B,C]
                #calc the likelihood
                if model.mode == "regression":
                    #in regression mode no softmax over output
                    #create one hot vectors R^10
                    one_hot_labels = F.one_hot(y, num_classes=model.output_dim).float()
                    #calc MSE
                    lhs = -0.5* F.mse_loss(output, one_hot_labels, reduction="mean") / ( 1**2) #assume sigma = 1 but maybe change alter
                else:
                    probs = output.gather(1, y.view(-1, 1)).squeeze() # index output to get [B]
                    log_probs = torch.log(probs + 1e-8) #for numeric stability
                    lhs = log_probs.mean()
                # calculate the KL div between prior and new var dist -> closed form since both mena field gaussian
                if prior is not None:
                    rhs = kl_div_gaussian_layer(model.get_var_dist(detach=False), prior)
                    rhs = rhs/len(curr_dataset)
                else:
                    rhs = 0
                loss = -lhs + rhs # - because we want to maximize ELBO so minimize negative elbo
                loss.backward()

                #Calc the uncertainty of the model
                model.eval()
                MI = model.get_MI(x,num_samples) #TODO consider evaluating before backprop
                mean_MI = MI.mean()
                model.train()

                current_mi.append(mean_MI.item())

                baseline_mean = 0
                baseline_std = 0

                if len(current_mi) == current_window_size:
                    #perform chekc only if the basline window is already full
                    if len(baseline_mi) == baseline_window_size:
                        baseline_mean = sum(baseline_mi) / len(baseline_mi)
                        baseline_std = (sum((x - baseline_mean) ** 2 for x in baseline_mi) / len(baseline_mi)) ** 0.5
                        current_mean = sum(current_mi) / len(current_mi)
                        means.append(baseline_mean)

                        if current_mean > baseline_mean + c * baseline_std:
                            print(f"Found a task switch at batch idx: {batch_idx} in epoch: {epoch}")
                            #model.add_head()
                            #optimizer = optim.Adam(model.parameters(), lr=lr)
                            baseline_mi.clear()
                            current_mi.clear()
                if current_mi:
                    # from current_mi -> baseline_mi if current_mi is full
                    baseline_mi.append(current_mi[0])
                #log metrics
                train_losses.append(loss.cpu().item())
                mutual_infos.append(mean_MI.cpu().item())
                means.append(baseline_mean)
                stds.append(baseline_std)
                optimizer.step()

        # Compute the final variational distribution (only used for prediction, and not propagation)
        #get q_t tilde before optimizing to get q_t

        acc, heads_chosen = evaluate_model(i, model, test_datasets, batch_size, device, num_samples)
        # Perform prediction at test input x*
        #preds = perform_predictions(model, curr_test_dataset,device)
        #collect intermediate results
        ret["acc"].append(acc.item())
        ret["train_losses"].extend(train_losses)
        ret["mutual_info"].extend(mutual_infos)
        ret["means"].extend(means)
        ret["stds"].extend(stds)
        #ret["heads_chosen"] = 
        
        print(f"Average acc of {acc} after training on task {i}")
        # init the model with the previous prior so we are not influenced by the coreset training
        #model.set_var_dist(prior)TODO reconsider the placement -> should not be used after last epocch
    #return final resutls
    return ret




