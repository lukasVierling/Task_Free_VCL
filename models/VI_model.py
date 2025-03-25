import torch
import torch.nn as nn
import torch.nn.functional as F

import math 

import copy

class BayesianLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        k = 1 / input_dim
        self.W_mu = nn.Parameter(torch.rand(input_dim, output_dim) * 2 * math.sqrt(k)-math.sqrt(k)) #TODO check if better when sampling form U(sqrt(k), sqrt(k))
        #self.W_mu = nn.Parameter(torch.empty(input_dim, hidden_dim))
        #nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        self.b_mu = nn.Parameter(torch.rand(output_dim) * 2 * math.sqrt(k)-math.sqrt(k))
        self.W_sigma = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1 - 6)
        self.b_sigma = nn.Parameter(torch.randn(output_dim) * 0.1- 6) #TODO Going form -3 to -7 increase acc by 10%!!
    
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
    def __init__(self, input_dim, output_dim, hidden_dim, mode="bernoulli", single_head=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mode = mode
        self.single_head = single_head
        if self.mode != "bernoulli" and self.mode !="regression":
            print("Please choose a valid mode for the model!")
        k = 1 / input_dim
        # first hidden layer -> fld layer in one parameter vector
        # Bayesian first hidden layer parameters (mean & log variance)
        self.W_mu = nn.Parameter(torch.rand(input_dim, hidden_dim) * 2 * math.sqrt(k)-math.sqrt(k)) #TODO check if better when sampling form U(sqrt(k), sqrt(k))
        #self.W_mu = nn.Parameter(torch.empty(input_dim, hidden_dim))
        #nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        self.b_mu = nn.Parameter(torch.rand(hidden_dim) * 2 * math.sqrt(k)-math.sqrt(k))
        self.W_sigma = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1 - 6.0)
        self.b_sigma = nn.Parameter(torch.randn(hidden_dim) * 0.1- 6.0) #TODO Going form -3 to -7 increase acc by 10%!!
        # second hidden layer, bayesian layer
        self.heads = nn.ModuleList()
        self.active_head = 0
        if self.single_head:
            print("Train on single head")
            self.add_head()

    def get_heads(self):
        heads = copy.deepcopy(self.heads)
        return heads
    
    def set_heads(self, heads):
        self.heads = copy.deepcopy(heads)
    
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
            
        var_dist["heads"] = []
        
        for head_idx in range(len(self.heads)):
            var_dist["heads"].append(self.heads[head_idx].get_var_dist(detach=detach))
            
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

        for head_idx, head_dist in enumerate(var_dist["heads"]):
            self.heads[head_idx].set_var_dist(head_dist)
    
    def add_head(self):
        '''
        add a new head to the model and set active head to this head
        '''
        # move the old head to the same device as previous head
        device = self.self.heads[-1].W_mu.device if len(self.heads) > 0 else self.W_mu.device
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

        if self.mode == "bernoulli":
            probs = F.softmax(y, dim=-1)
        elif self.mode=="regression":
            probs = y
        return probs


class GenerativeModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        k = 1 / hidden_dim
        # first hidden layer -> fld layer in one parameter vector
        # Bayesian first hidden layer parameters (mean & log variance)
        self.last_layer = torch.nn.Linear(hidden_dim,output_dim)
        self.W_mu = nn.Parameter(torch.rand(hidden_dim, output_dim) * 2 * math.sqrt(k)-math.sqrt(k)) #TODO check if better when sampling form U(sqrt(k), sqrt(k))
        #nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        scale = 0.1
        shift = 7.0
        self.b_mu = nn.Parameter(torch.rand(output_dim) * 2 * math.sqrt(k)-math.sqrt(k))
        self.W_sigma = nn.Parameter(torch.randn(hidden_dim, output_dim) * scale - shift)
        self.b_sigma = nn.Parameter(torch.randn(output_dim) * scale- shift) #TODO Going form -3 to -7 increase acc by 10%!!

        print(f"Initialize a model with Variance of {math.exp(-shift)}"  )
        # second hidden layer, bayesian layer

        #self.W_sigma = torch.nn.Parameter(torch.full((hidden_dim, output_dim), 1e-6))
        #self.b_sigma = torch.nn.Parameter(torch.full((output_dim,), 1e-6))

        #Individual Encoder for every task
        self.encoders = nn.ModuleList()
        self.active_encoder = 0
        
        #Inidividual Heads
        self.heads = nn.ModuleList()
        self.active_head = 0
    
    def get_encoders(self):
        encoders = copy.deepcopy(self.encoders)
        return encoders

    def get_heads(self):
        heads = copy.deepcopy(self.heads)
        return heads
    
    def set_heads(self, heads):
        self.heads = copy.deepcopy(heads)
    
    def set_encoders(self, encoders):
        self.encoders = copy.deepcopy(encoders)
    
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

    
    def add_head(self):
        '''
        add a new head to the model and set active head to this head
        '''
        if self.single_head and len(self.heads) > 0:
            print(" ---------- \n\n\nWarning! Can't add more heads in single head mode! \n\n\n ---------- \n\n\n")
        # move the old head to the same device as previous head
        device = self.heads[-1].weight.device if len(self.heads) > 0 else self.W_mu.device
        new_head = nn.Linear(self.latent_dim, self.hidden_dim).to(device)
        self.heads.append(new_head)
        self.active_head = len(self.heads)-1
        print("Added new head, current head index: ", self.active_head)

    def add_encoder(self):
        device = self.encoders[-1][0].weight.device if len(self.encoders) > 0 else self.W_mu.device
        new_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2* self.latent_dim) #to predict mean and log(sigma^2)
        ).to(device)
        self.encoders.append(new_encoder)
        self.active_encoder = len(self.encoders)-1
        print("Added new encoder, current encoder index: ", self.active_encoder)
        

    def activate_head(self, i):
        '''
        activate one of the model's heads
        '''
        if 0 <= i < len(self.heads):
            print("Change active head to: ",i)
            self.active_head = i
        else:
            print("Head index out of bounds, active head:", self.active_head)

    def activate_encoder(self, i):
        '''
        activate one of the model's encoder
        '''
        if 0 <= i < len(self.encoders):
            print("Change active encoder to: ",i)
            self.active_encoder = i
        else:
            print("Encoder index out of bounds, active encoder:", self.active_encoder)

    def get_active_head_idx(self):
        return self.active_head

    def get_active_encoder_idx(self):
        return self.active_encoder

    def encode(self, x):
        #get bs
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # fold the encoder into a layer

        #encoder the image
        params = self.encoders[self.active_encoder](x)

        mean = params[:, :self.latent_dim] #skip batch dim
        log_var = params[:, self.latent_dim:]

        #eps = torch.randn_like(mean)

        #z = mean + torch.exp(log_var * 0.5) * eps #0.5 because we need std not var

        return mean, log_var
    
    def decode(self,z):
        # use individual head

        h = F.relu(self.heads[self.active_head](z))

        W_epsilon = torch.randn_like(self.W_sigma) #randn for normal dist (0,1) default
        b_epsilon = torch.randn_like(self.b_sigma)

        W = self.W_mu + torch.exp(0.5*self.W_sigma) * W_epsilon
        b = self.b_mu + torch.exp(0.5*self.b_sigma) * b_epsilon
        
        # forward through shared layer
        y = h @ W  + b

        #values between 0 and 1
        normalized_y = F.sigmoid(y)

        #normalized_y = F.sigmoid(self.last_layer(h))

        return normalized_y



    def forward(self, x):
        #get bs
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # fold the encoder into a layer

        #encoder the image
        params = self.encoders[self.active_encoder](x)

        mean = params[:, :self.latent_dim] #skip batch dim
        log_var = params[:, self.latent_dim:]

        eps = torch.randn_like(mean)

        z = mean + torch.exp(log_var * 0.5) * eps #0.5 because we need std not var

        # use individual head

        h = F.relu(self.heads[self.active_head](z))

        W_epsilon = torch.randn_like(self.W_sigma) #randn for normal dist (0,1) default
        b_epsilon = torch.randn_like(self.b_sigma)

        W = self.W_mu + torch.exp(0.5*self.W_sigma) * W_epsilon
        b = self.b_mu + torch.exp(0.5*self.b_sigma) * b_epsilon
        
        # forward through shared layer
        y = h @ W  + b

        #values between 0 and 1
        normalized_y = F.sigmoid(y)


        #normalized_y = F.sigmoid(self.last_layer(h))

        return normalized_y, mean, log_var
    

