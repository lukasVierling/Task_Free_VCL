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
import torch.optim as optim
# my imports
from utils.utils import kl_div_gaussian

def laplace_reg(model,fisher_diags, phis):
    #calculate the laplace loss
    #theta are current stacked parameters
    theta = model.get_stacked_params(detach=False)
    #clac the loss sum
    loss = 0
    #TODO consdier eval on subset for FIM
    for idx,phi in enumerate(phis.values()):
        delta = theta-phi
        fisher_diag = fisher_diags[idx]
        loss += 0.5 * delta.T @ (delta * fisher_diag)
    # bec. we use diagonal matrices we can just do elementwise multiplication at the end
    #TODO consider a mean
    #print(f"Laplace regularization loss of: {loss}")
    return loss

def vae_loss(recon_x, x, mean, log_var):
    #eps = 1e-6 #for stability
    #likelihood part
    bernoulli_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #now the KL part
    kl = - 0.5 * torch.sum(1+log_var - mean**2 - torch.exp(log_var))
    return bernoulli_loss + kl #both terms are positive but then flip sign later


def train_one_task(model, fisher_diags, phis, curr_dataset, train_datasets ,batch_size, epochs, lr, device):
    data_loader = DataLoader(curr_dataset, shuffle=True, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in tqdm(range(epochs), desc=f"Training for {epochs} epochs"):
        for x,y in tqdm(data_loader, desc=f"Train epoch {epoch}"):
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            model.zero_grad()
            #forward through the model to get likelihood
            output, mean, log_var = model(x) #-> returns [B,C]
            #calc the VAE loss
            
            lhs = vae_loss(output, x, mean, log_var) #this is mean!
            # calculate the KL div between prior and new var dist -> closed form since both mena field gaussian
            if phis:
                rhs = laplace_reg(model, fisher_diags, phis) * batch_size
                #rhs = 0
            else:
                rhs = 0
            loss = lhs + rhs # - because we want to maximize ELBO so minimize negative elbo TODO chck if implemented correct?
            loss.backward()
            optimizer.step()

def evaluate_on_all_tasks(task_idx, model, classifier, test_datasets, batch_size, epochs, lr, device):
    accs = []
    #get the currently activated head
    prev_head = model.get_active_head_idx()
    prev_encoder = model.get_active_encoder_idx()
    uncertainties = []
    llhs = []
    for head_idx, test_dataset in enumerate(test_datasets[:task_idx+1]):
        #excluseive 
        model.activate_head(head_idx)
        model.activate_encoder(head_idx)
        #append the accuracy
        uncertainty, llh = perform_generations(model, classifier, test_dataset, batch_size, device)
        uncertainties.append(uncertainty)
        llhs.append(llh)
        print(f"Got metrics of: ",llhs[-1],uncertainties[-1])
    #activate previous head
    model.activate_head(prev_head)
    model.activate_encoder(prev_encoder)
    #return the accuracy
    return uncertainties, llhs


def perform_generations(model, classifier, curr_test_dataset,batch_size,device, num_samples=5000):
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
        flat_images = model.decode(z).view(num_samples, 1, 28, 28)
        #classify the images
        probs = classifier.get_probs(flat_images)
        #calc the KL div with the one-hot vector
        one_hot = F.one_hot(torch.full((num_samples,), label, device=device), num_classes=10).float() #hardcode 10 for mnist
        kl_div = F.kl_div(torch.log(probs), one_hot, reduction="mean") #expects log probabilities
        uncertainty_measure = kl_div

        #sample performance
        sample_generations(model, None, curr_test_dataset, batch_size, device)

        ###
        #llh part
        ###
    
        #go over test set
        summed_ll = 0
        test_loader = DataLoader(curr_test_dataset, shuffle=False, batch_size = 1)
        K = num_samples
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

            if torch.isnan(log_q_z).any():
                print("NaN detected in log_q_z!")

            if torch.isnan(log_p_z).any():
                print("NaN detected in log_p_z!")
                
            #obtain log likelihood
            img_mean = model.decode(z)
            
            #numeric stability prevent NaN
            img_mean = model.decode(z)
            #stability
            eps = 1e-7 
            img_mean = img_mean.clamp(eps, 1 - eps)
            log_p_x_z = x * torch.log(img_mean) + (1-x) * torch.log(1-img_mean)
            log_p_x_z = log_p_x_z.view(batch_size*K, -1).sum(dim=1)

            if torch.isnan(log_p_x_z).any():
                print("NaN detected in log_p_x_z!")

            log_p_x = log_p_x_z + log_p_z - log_q_z

            log_p_x = log_p_x.view(K, batch_size)

            #find maximum element for every "batch" of K elements
            max_log_w, _ = torch.max(log_p_x, dim=0, keepdim=True)
            shifted = log_p_x - max_log_w
            results = max_log_w + torch.log(torch.mean(torch.exp(shifted), dim=0))

            if torch.isnan(results).any():
                print("NaN detected in results!")

            summed_ll += results.sum()

        average_ll = summed_ll / len(curr_test_dataset)

    return uncertainty_measure, average_ll

def sample_generations(model, classifier, curr_test_dataset, batch_size,device):
    #make a test to visually verify if it works
    z = torch.randn(25, 50).to(device)
    #x,y = curr_test_dataset[0]
    #print(x.shape)
    #mean, log_var = model.encode(x.unsqueeze(0).to(device))
    #z = mean + z * torch.exp(0.5*log_var)
    # Decode to images
    with torch.no_grad():
        generated = model.decode(z)  # Output: [n_images, 784]
        generated = generated.view(-1, 28, 28).cpu()  # Reshape to [n_images, 28, 28]

    # Plot images in a 5x5 grid
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    for i, ax in enumerate(axs.flat):
        ax.imshow(generated[i], cmap='gray')
        ax.axis('off')

        plt.tight_layout()
    save_dir = "outputs"
    filename = None
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ewc_generation_{timestamp}.png"

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved generations to: {save_path}")


def ewc(model, train_datasets, test_datasets, classifier, batch_size, epochs, lr, lambdas, device="cpu"):
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
    #init with covariance of gaussian prior TODO implement
    fisher_diags = {}
    thetas = {} #start with MLE 
    # get the number of datasets T
    T = len(train_datasets)
    for i in tqdm(range(T), desc="Training on tasks..."):
        #add task specific head to the model
        model.add_head()
        model.add_encoder()
        # get the current dataset D_i and train set
        curr_dataset = train_datasets[i]
        curr_test_dataset = test_datasets[i]
        # update the coreset with D_i
        # Update the variational distribution for non-coreset data points
        train_one_task(model, fisher_diags, thetas, curr_dataset, train_datasets ,batch_size, epochs, lr, device)
        #get FIM of current model and the parameters for next loss
        fisher_diags[i] = model.get_fisher(curr_dataset).to(device) * lambdas[i] # directly apply the scaling
        
        if not(use_regularization):
            thetas = None
            print("Not using regularization")
        else:
            thetas[i] = model.get_stacked_params(detach=True).to(device) #just for computing next loss no gradients on those
            print(f"Saved new theta of size: {thetas[i].shape}")
        
        print(f"Saved new fisher_diagonal sum of size: {fisher_diags[i].shape}")
        # evaluate the performance on all tasks
        results = evaluate_on_all_tasks(i, model, classifier, test_datasets, batch_size, epochs, lr, device)
        # Perform prediction at test input x*
        #preds = perform_predictions(model, curr_test_dataset,device)
        #collect intermediate results
        ret.append(results)
        print(f"Results of {results} after training on task {i}")
        # init the model with the previous prior so we are not influenced by the coreset training
        #model.set_var_dist(prior)TODO reconsider the placement -> should not be used after last epocch
    #return final resutls
    return ret