import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def laplace_reg(model, hessian_diag, phi):
    theta = model.get_stacked_params(detach=False)

    loss = 0
    delta = theta - phi
    #diff to EWC: only difference between previous task and current 
    # other diff: we don't have lambda hyperparameters
    loss = 0.5 * delta.T @ (delta * hessian_diag)

    return loss


def train_one_task(model, hessian_diag, phi, lambd, curr_dataset, train_datasets ,batch_size, epochs, lr, device):
    data_loader = DataLoader(curr_dataset, shuffle=True, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr)
    #print("prev theta:" ,torch.sum(phi))
    #print("hessian diag:", torch.sum(hessian_diag))
    for epoch in tqdm(range(epochs), desc=f"Training for {epochs} epochs"):
        for x,y in tqdm(data_loader, desc=f"Train epoch {epoch}"):
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            #forward through the model to get likelihood
            output = model(x) #-> returns [B,C]
            #calc the likelihood
            probs = output.gather(1, y.view(-1, 1)).squeeze() # index output to get [B]
            log_probs = torch.log(probs + 1e-8) #for numeric stability
            lhs = log_probs.mean()
            # calculate the KL div between prior and new var dist -> closed form since both mena field gaussian
            if phi is not None:
                rhs = lambd * laplace_reg(model, hessian_diag, phi)
                #rhs = 0
                #rhs = rhs / len(curr_dataset)
                #print("RHS loss:", rhs)
            else:
                rhs = 0
            #print("Loss lhs:", lhs)
            #print("Loss rhs:", rhs)
            loss = -lhs + rhs # - because we want to maximize ELBO so minimize negative elbo TODO chck if implemented correct? #TODO consider normalizing rhs
            loss.backward()
            optimizer.step()

def evaluate_on_all_tasks(task_idx, model, test_datasets, batch_size, epochs, lr, device):
    accs = []
    #get the currently activated head
    prev_head = model.get_active_head_idx()
    for head_idx, test_dataset in enumerate(test_datasets[:task_idx+1]):
        #excluseive 
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


def lp(model, train_datasets, test_datasets, batch_size, epochs, lr, lambd, device="cpu"):
    '''
    Input:
    - prior : prior distribution
    - dataset : list of datasets
    Output: 
    - [(q_t,p_t)] t=1,...,T : Variational and predictive distribution at each step
    '''
    use_regularization = True #TODO remove this 
    model.to(device)
    ret = []
    #init with covariance of gaussian prior
    hessian_diag = torch.zeros_like(model.get_stacked_params(detach=True)).to(device) #TODO what is the properi nitialization?
    prev_theta = None #start with MLE 
    # get the number of datasets T
    T = len(train_datasets)
    for i in tqdm(range(T), desc="Training on tasks..."):
        #add task specific head to the model
        model.add_head()
        # get the current dataset D_i and train set
        curr_dataset = train_datasets[i]
        curr_test_dataset = test_datasets[i]
        # update the coreset with D_i
        # Update the variational distribution for non-coreset data points
        train_one_task(model, hessian_diag, prev_theta, lambd, curr_dataset, train_datasets ,batch_size, epochs, lr, device)
        #get FIM of current model and the parameters for next loss
        hessian_diag += model.get_hessian(curr_dataset).to(device) # directly apply the scaling
        #print("Use Fisher approximation instead of real Hessian")#TODO adjust when chagned
        if not(use_regularization):
            prev_theta = None
            print("Not using regularization")
        else:
            prev_theta = model.get_stacked_params(detach=True).to(device) #just for computing next loss no gradients on those
            print(f"Saved new theta of size: {prev_theta.shape}")
        
        print(f"Saved new hessian daigonal of size: {hessian_diag.shape}")
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