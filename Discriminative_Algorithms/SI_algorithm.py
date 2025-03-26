import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def contribution_loss(model, prev_theta, omega):
    theta = model.get_stacked_params(detach=False) #calc gradients on it
    squared_diff = (theta-prev_theta)**2
    loss = omega * squared_diff
    #print("Loss shape: ", loss.shape)
    loss = loss.sum() #TODO maybe sum??? sum got 93->88
    #print("after sum",loss.shape)
    return loss

def train_one_task(model, omega, prev_theta, c, curr_dataset, train_datasets ,batch_size, epochs, lr, device):
    data_loader = DataLoader(curr_dataset, shuffle=True, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr)
    prev_params = model.get_stacked_params(detach=True)
    contribution = torch.zeros_like(prev_params)
    for epoch in tqdm(range(epochs), desc=f"Training for {epochs} epochs"):
        for x,y in tqdm(data_loader, desc=f"Train epoch {epoch}"):
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            #forward through the model to get likelihood
            output = model(x) #-> returns [B,C]
            #calc the likelihood
            probs = output.gather(1, y.view(-1, 1)).squeeze() # index output to get [B]
            log_probs = torch.log(probs + 1e-8) #for numeric stability
            lhs = log_probs.mean() #TODO somehow mean and sum got almost same results lol
            # calculate the KL div between prior and new var dist -> closed form since both mena field gaussian
            if omega is not None:
                #rhs = 0
                rhs = c * contribution_loss(model, prev_theta, omega) #mult with c constant
            else:
                rhs = 0
            loss = -lhs + rhs # - because we want to maximize ELBO so minimize negative elbo TODO chck if implemented correct?

            loss.backward()
            optimizer.step()

            #update contribution
            grads = model.get_stacked_gradients()
            params = model.get_stacked_params(detach=True)
            diff = params - prev_params
            contribution += -grads * diff

            prev_params = params

    return contribution

def evaluate_on_all_tasks(task_idx, model, test_datasets, batch_size, epochs, lr, device):
    accs = []
    #get the currently activated head
    prev_head = model.get_active_head_idx()
    for head_idx, test_dataset in enumerate(test_datasets[:task_idx+1]):
        #excluseive 
        if not(model.single_head):
            model.activate_head(head_idx)
        #append the accuracy
        accs.append(perform_predictions(model, test_dataset, batch_size, device))
        print(f"Got accuracy of {accs[-1]*100}%")
    #activate previous head
    model.activate_head(prev_head)
    #return the accuracy
    return sum(accs) / len(accs)


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


def si(model, train_datasets, test_datasets, batch_size, epochs, lr, damping_param, c, device="cpu"):
    '''
    Input:
    - prior : prior distribution
    - dataset : list of datasets
    Output: 
    - [(q_t,p_t)] t=1,...,T : Variational and predictive distribution at each step
    '''
    model.to(device)
    ret = []
    #init with covariance of gaussian prior TODO implement
    #theta and delta saved to calc the next omega
    prev_theta = model.get_stacked_params(detach=True)#TODO probably get params
    delta = None
    #small omega
    contribution =None
    #save the Omega
    omega = None
    # get the number of datasets T
    T = len(train_datasets)
    for i in tqdm(range(T), desc="Training on tasks..."):
        #add task specific head to the model
        if not(model.single_head):
            model.add_head()
        # get the current dataset D_i and train set
        curr_dataset = train_datasets[i]
        curr_test_dataset = test_datasets[i]
        # update the coreset with D_i
        # Update the variational distribution for non-coreset data points
        contribution = train_one_task(model, omega, prev_theta, c, curr_dataset, train_datasets ,batch_size, epochs, lr, device)
        #calc the new theta
        theta = model.get_stacked_params(detach=True) #TODO probably get params
        #calc new delta
        delta = prev_theta - theta
        # calc new omega
        if omega is None:
            omega = torch.zeros_like(contribution)
        omega += contribution / (delta**2 + damping_param)
        #set prev_theta to current theta
        prev_theta = theta
        
        # evaluate the performance on all tasks
        acc = evaluate_on_all_tasks(i, model, test_datasets, batch_size, epochs, lr, device)
        # Perform prediction at test input x*
        #preds = perform_predictions(model, curr_test_dataset,device)
        #collect intermediate results
        ret.append(acc)
        print(f"Average acc of {acc} after training on task {i}")
        # init the model with the previous prior so we are not influenced by the coreset training
        #model.set_var_dist(prior)TODO reconsider the placement -> should not be used after last epocch
    #return final resutls
    return ret