import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy


class BayesianLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        k = 1 / input_dim
        self.W_mu = nn.Parameter(torch.rand(input_dim, output_dim) * 2 * math.sqrt(k)-math.sqrt(k))
        self.b_mu = nn.Parameter(torch.rand(output_dim) * 2 * math.sqrt(k)-math.sqrt(k))
        self.W_sigma = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1 - 6) #note this is actually the log variance
        self.b_sigma = nn.Parameter(torch.randn(output_dim) * 0.1- 6) 

        #nn.init.xavier_normal_(self.W_mu)
        #self.W_sigma.data.fill_(-5.0)
        #nn.init.zeros_(self.b_mu)
        #self.b_sigma.data.fill_(-5.0)
    
    def get_var_dist(self, detach=True):
        if detach:
            var_dist = {"W_mu": self.W_mu.detach().clone(), 
                        "W_sigma": self.W_sigma.detach().clone(), 
                        "b_mu":self.b_mu.detach().clone(), 
                        "b_sigma": self.b_sigma.detach().clone()
                        }
        else:
            var_dist = {"W_mu": self.W_mu, 
                        "W_sigma": self.W_sigma, 
                        "b_mu":self.b_mu, 
                        "b_sigma": self.b_sigma
                        }
        return var_dist
    
    def set_var_dist(self, var_dist):
        #get  current device
        device = self.W_mu.device
        # detach them from any comp graph and trainable through nn.param
        print("Initialize the shared layer with new var_dist")
        self.W_mu = torch.nn.Parameter(var_dist["W_mu"].detach().clone().to(device))
        self.b_mu = torch.nn.Parameter(var_dist["b_mu"].detach().clone().to(device))
        self.W_sigma = torch.nn.Parameter(var_dist["W_sigma"].detach().clone().to(device))
        self.b_sigma = torch.nn.Parameter(var_dist["b_sigma"].detach().clone().to(device))  
        
    
    def forward(self, x):
        #get bs
        # fold the encoder into a layer

        W_epsilon = torch.randn_like(self.W_sigma) #randn for normal dist (0,1) default
        b_epsilon = torch.randn_like(self.b_sigma)

        W = self.W_mu + torch.exp(0.5*self.W_sigma) * W_epsilon
        b = self.b_mu + torch.exp(0.5*self.b_sigma) * b_epsilon
        
        # forward through first layer
        z = x @ W  + b

        return z

class DiscriminativeModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, mode="bernoulli", single_head=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mode = mode
        self.single_head = single_head
        self.encoder = BayesianLayer(input_dim, hidden_dim)
        self.heads = nn.ModuleList()
        self.active_head = 0
        if self.single_head:
            print("Train on single head")
            self.add_head()
        print(f"Params:\n   input_dim: {input_dim}\n    hidden_dim:{hidden_dim}\n   output_dim:{output_dim}\n    mode:{mode}\n    single_head:{single_head}")

    def get_heads(self, detach=True):
        heads = []
        for head_idx in range(len(self.heads)):
            heads.append(self.heads[head_idx].get_var_dist(detach))
        return heads

    
    def set_heads(self, heads):
        for head_idx in range(len(self.heads)):
            heads.append(self.heads[head_idx].set_var_dist(heads[head_idx]))
    
    def get_var_dist(self, detach=True):
        return self.encoder.get_var_dist(detach)

    def set_var_dist(self, var_dist):
        #get  current device
        self.encoder.set_var_dist(var_dist)
    
    def add_head(self):
        '''
        add a new head to the model and set active head to this head
        '''
        # move the old head to the same device as previous head
        device = self.heads[-1].W_mu.device if len(self.heads) > 0 else self.encoder.W_mu.device
        new_head = BayesianLayer(self.hidden_dim, self.output_dim).to(device)
        self.heads.append(new_head)
        self.active_head = len(self.heads)-1
        print("Added new head, current head index: ", self.active_head)

    def activate_head(self, i):
        '''
        activate one of the model's heads
        '''
        if 0 <= i < len(self.heads):
            print("Change active head to: ",i)
            self.active_head = i
        else:
            print("Head index out of bounds, active head:", self.active_head)

    def get_active_head_idx(self):
        return self.active_head
    
    def get_MI(self, x, num_samples=25, return_predictions=False, head_idx=None, var=0.01, eps=1e-8):
        #x should be in batch form ( if one sample then (1,D))
        #calc the entropy
        head_idx = self.active_head if head_idx is None else head_idx

        if self.mode == "bernoulli":
            batch_size = x.shape[0]
            with torch.no_grad():
                ys = [self(x.view(batch_size, -1), head_idx=head_idx) for _ in range(num_samples)]
                ys = torch.stack(ys, dim=0)
                y = ys.view(num_samples, batch_size, -1) # -> [num_samp, B, D]
                avg_pred = y.mean(dim=0) #-> reduce over num_samp and get [B,D]
                avg_pred_entropy = -torch.sum(avg_pred * torch.log(avg_pred +eps), dim=-1)
                entropies = -torch.sum(y * torch.log(y + eps), dim=-1)
                avg_entropy = torch.mean(entropies, dim=0)
                MI = avg_pred_entropy - avg_entropy
            if not return_predictions:
                return MI
            else:
                return MI, avg_pred
        
        
    def forward_with_routing(self, x, routing_mode="batchwise", calculation_mode="sampling", num_samples=25, var=0.01):
        if routing_mode != "batchwise":
            print("Please use batchwise mode, other modes not impelemented")
        if routing_mode == "batchwise":
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

        return best_preds, best_head



    def forward(self, x, head_idx=None):
        #get bs
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # fold the encoder into a layer

        z = F.relu(self.encoder(x))

        # choose head
        if head_idx is None:
            head_idx = self.active_head

        #forward through head
        y = self.heads[head_idx](z)

        probs = F.softmax(y, dim=-1)
        return probs
                