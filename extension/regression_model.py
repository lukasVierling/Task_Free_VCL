import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        k = 1 / input_dim
        
        self.W_mu = nn.Parameter(torch.rand(output_dim,input_dim) * 2 * math.sqrt(k)-math.sqrt(k)) #TODO check if better when sampling form U(sqrt(k), sqrt(k))
        self.b_mu = nn.Parameter(torch.rand(output_dim) * 2 * math.sqrt(k)-math.sqrt(k))
        self.W_log_var = nn.Parameter(torch.randn(output_dim,input_dim) * 0.1 - 6) #note this is actually the log variance
        self.b_log_var = nn.Parameter(torch.randn(output_dim) * 0.1- 6) #TODO Going form -3 to -7 increase acc by 10%!!
        '''
        self.W_mu = nn.Parameter(torch.empty(output_dim, input_dim).normal_(0, 0.01)) #TODO check if better when sampling form U(sqrt(k), sqrt(k))
        self.b_mu = nn.Parameter(torch.empty(output_dim,).normal_(0, 0.01))
        self.W_log_var = nn.Parameter(torch.full((output_dim, input_dim), -5.0)) #note this is actually the log variance
        self.b_log_var = nn.Parameter(torch.full((output_dim, ), -5.0)) #TODO Going form -3 to -7 increase acc by 10%!!
        '''
        self.prior_W_mu = None
        self.prior_W_log_var = None
        self.prior_b_mu= None
        self.prior_b_log_var = None

       # nn.init.xavier_normal_(self.W_mu)
        #self.W_sigma.data.fill_(-5.0)
        #nn.init.zeros_(self.b_mu)
        #self.b_sigma.data.fill_(-5.0)
    def update_prior(self):
        #update the prior of this layer
        self.prior_W_mu = self.W_mu.clone().detach()
        self.prior_W_log_var = self.W_log_var.clone().detach()
        self.prior_b_mu = self.b_mu.clone().detach()
        self.prior_b_log_var = self.b_log_var.clone().detach()
    
    def forward(self, x, sample=True):
        #get bs
        # fold the encoder into a layer
        if sample:
            W_epsilon = torch.randn_like(self.W_log_var) #randn for normal dist (0,1) default
            b_epsilon = torch.randn_like(self.b_log_var)

            W = self.W_mu + torch.exp(0.5*self.W_log_var) * W_epsilon
            b = self.b_mu + torch.exp(0.5*self.b_log_var) * b_epsilon
        else:
            W = self.W_mu
            b = self.b_mu
        # forward through first layer
        z = F.linear(x, W, b)
        return z

    def kl_divergence(self):
        if (self.prior_W_mu is None or self.prior_W_log_var is None or
            self.prior_b_mu is None or self.prior_b_log_var is None):
            return 0
        
        log_var_1 = self.W_log_var
        mu_1 = self.W_mu
        log_var_2 = self.prior_W_log_var
        mu_2 = self.prior_W_mu

        #flatten into vectors
        log_var_1 = log_var_1.reshape(-1)
        mu_1 = mu_1.reshape(-1)
        log_var_2 = log_var_2.reshape(-1)
        mu_2 = mu_2.reshape(-1)

        #transform Sigma_1 form log var into std
        Sigma_1 = torch.exp(log_var_1)
        Sigma_2 = torch.exp(log_var_2)

        #calc the KL divergence
        trace_part = torch.sum(Sigma_1 / Sigma_2) #use that matrices are diagonal so we can just divide
        squared_part = ((mu_1-mu_2)/Sigma_2).T @ (mu_1 -mu_2)
        d = Sigma_1.shape[0] #dimension is the lenght for squared matrices
        log_part = torch.sum(torch.log(Sigma_2)-torch.log(Sigma_1))

        weights_loss = 0.5*(trace_part + squared_part - d + log_part)

        ###
        #bias
        ####

        log_var_1 = self.b_log_var
        mu_1 = self.b_mu
        log_var_2 = self.prior_b_log_var
        mu_2 = self.prior_b_mu

        #flatten into vectors
        log_var_1 = log_var_1.view(-1)
        mu_1 = mu_1.view(-1)
        log_var_2 = log_var_2.view(-1)
        mu_2 = mu_2.view(-1)

        #transform Sigma_1 form log var into std
        Sigma_1 = torch.exp(log_var_1)
        Sigma_2 = torch.exp(log_var_2)


        #calc the KL divergence
        trace_part = torch.sum(Sigma_1 / Sigma_2) #use that matrices are diagonal so we can just divide
        squared_part = ((mu_1-mu_2)/Sigma_2).T @ (mu_1 -mu_2)
        d = Sigma_1.shape[0] #dimension is the lenght for squared matrices
        log_part = torch.sum(torch.log(Sigma_2)-torch.log(Sigma_1))

        bias_loss = 0.5*(trace_part + squared_part - d + log_part)

        return weights_loss + bias_loss


class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mode="bernoulli", single_head=False):
        super().__init__()
        self.encoder = BayesianLinear(input_dim, hidden_dim)
        self.heads = nn.ModuleList()
        self.output_dim = output_dim
        self.active_head = 0
        self.mode = mode
        self.single_head = single_head
        print(f"Params:\n   input_dim: {input_dim}\n    hidden_dim:{hidden_dim}\n   output_dim:{output_dim}\n    mode:{mode}\n    single_head:{single_head}")
    
    def get_active_head_idx(self):
        return self.active_head
    
    def activate_head(self, head_idx):
        self.active_head = head_idx

    def add_head(self):
        if len(self.heads) > 1:
            #don't update encoder prior for the initial head
            self.encoder.update_prior()
            if self.single_head:
                self.heads[-1].update_prior()
        self.heads.append(BayesianLinear(self.encoder.output_dim, self.output_dim).to(self.encoder.W_mu.device))
        self.active_head = len(self.heads)-1
    
    def forward_with_routing(self, x, routing_mode="batchwise", calculation_mode="sampling", num_samples=25, var=0.01):
        if routing_mode != "batchwise":
            print("Please use batchwise mode, other modes not impelemented")
        if routing_mode == "batchwise":
            if calculation_mode == "sampling":
                MIs = []
                preds = []
                best_head = None
                for head_idx in range(len(self.heads)):
                    MI, avg_pred = self.get_MI(x, num_samples=num_samples, return_predictions=True, head_idx=head_idx)
                    avg_MI = MI.mean(dim=0)
                    MIs.append(avg_MI)
                    preds.append(avg_pred)
                MIs = torch.stack(MIs)
                best_head =  int(torch.argmin(MIs).item())
                best_preds = preds[best_head]

            if calculation_mode == "closed_form":
                MIs = []
                preds = []
                best_head = None
                #print("heads", len(self.heads))
                for head_idx in range(len(self.heads)):
                    MI = self.calc_MI(x, sigma2=var, head_idx=head_idx)
                 #   print("MI shape:", MI.shape)
                    avg_MI = MI.mean()
                    MIs.append(avg_MI)
                MIs = torch.stack(MIs)
                #print("MIs shape:",MIs.shape)
                best_head =  int(torch.argmin(MIs).item())
                batch_size = x.shape[0]
                #print("best head: ", best_head)
                preds = [self(x.view(batch_size, -1), head_idx=best_head) for _ in range(num_samples)]

                best_preds = torch.stack(preds)
                #print("best predas shape:", best_preds.shape)
                best_preds = best_preds.mean(dim=0)
                #print("best predas shape:", best_preds.shape)

        return best_preds, best_head

    
    def forward(self, x, head_idx=None, sample=True):
        if head_idx is None:
            head_idx = self.active_head
        z = F.relu(self.encoder(x, sample=sample))
        y = self.heads[head_idx](z, sample=sample)
        return y
    
    def kl_divergence(self):
        if self.single_head:
            kl_div = self.encoder.kl_divegence() + self.heads[-1].kl_divergence()
        else:
            kl_div = self.encoder.kl_divergence()
        return kl_div
    
    def calc_MI(self, x, sigma2=0.01, head_idx=None, eps=1e-8):
        # get the head 
        if head_idx is None:
            head_idx = self.active_head
        z = F.relu(self.encoder(x, sample=False))
        # get current ehad
        head = self.heads[head_idx]
        # get Sigma of head
        weight_sigma = torch.exp(0.5 * head.W_log_var) 
        # get Sigma of head bias
        bias_sigma = torch.exp(0.5 * head.b_log_var)
        # calc the z^T Sigma z term
        var_term = torch.matmul(z**2, (weight_sigma**2).T) 
        # include the bias
        var_term = var_term + bias_sigma**2
        # calc the closed form solution
        mi = 0.5 * torch.log(1 + (var_term + eps) / sigma2)
        #print("mi shape:", mi.shape)
        return mi 


    def get_MI(self, x, num_samples=50, return_predictions=True, head_idx=None, sigma2=0.01, eps=1e-8):
        #set head_idx
        if head_idx is None:
            head_idx = self.active_head
        self.eval()
        with torch.no_grad():
            # sample num_samples times
            y_samples = [self(x, head_idx, sample=True) for _ in range(num_samples)]
            y_samples = torch.stack(y_samples)
            #calc the exepcted value over weight distribution (monte carlo integral)
            pred_var = y_samples.var(dim=0).mean(dim=1)
            # use closed form solution for nomral dist with approx VAriance
            mi = 0.5 * torch.log(1 + (pred_var + eps) / sigma2)
            return mi
