import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminativeModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # first hidden layer -> fld layer in one parameter vector
        # Bayesian first hidden layer parameters (mean & log variance)
        self.W_mu = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.b_mu = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.W_sigma = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1 -3.0)
        self.b_sigma = nn.Parameter(torch.randn(hidden_dim) * 0.1-3.0) #TODO maybe init W as 0 and sigma with 1 
        # second hidden layer, bayesian layer
        self.heads = nn.ModuleList()
        self.active_head = 0
    
    def get_var_dist(self):
        var_dist = {"W_mu": self.W_mu.detach().clone(), 
                    "W_sigma": self.W_sigma.detach().clone(), 
                    "b_mu":self.b_mu.detach().clone(), 
                    "b_sigma": self.b_sigma.detach().clone()
                    }
        return var_dist
    
    def add_head(self):
        '''
        add a new head to the model and set active head to this head
        '''
        new_head = nn.Linear(self.hidden_dim, self.output_dim)
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

    def forward(self, x):
        #get bs
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # fold the encoder into a layer

        W_epsilon = torch.randn_like(self.W_sigma) #randn for normal dist (0,1) default
        b_epsilon = torch.randn_like(self.b_sigma)

        W = self.W_mu + torch.exp(0.5*self.W_sigma) * W_epsilon
        b = self.b_mu + torch.exp(0.5*self.b_sigma) * b_epsilon
        
        # forward through first layer
        z = F.relu(x @ W  + b)
        # forward throught head
        y = self.heads[self.active_head](z)
        probs = F.softmax(y, dim=-1)
        return probs
