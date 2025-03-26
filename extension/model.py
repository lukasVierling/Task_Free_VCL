import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, input_dim, output_dim, hidden_dim, mode="bernoulli", single_head=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mode = mode
        self.single_head = single_head
        if self.mode != "bernoulli" and self.mode !="regression":
            print("Please choose a valid mode for the model!")
        self.encoder = BayesianLayer(input_dim, hidden_dim)
        self.heads = nn.ModuleList()
        self.active_head = 0
        if self.single_head:
            print("Train on single head")
            self.add_head()
    
    def get_MI(self, x, num_samples=25, return_predictions=False, head_idx=None):
        #x should be in batch form ( if one sample then (1,D))
        #calc the entropy
        batch_size = x.shape[0]
        with torch.no_grad():
            x = x.repeat(num_samples, 1, 1) # -> [num_samp, B, D]
            x = x.view(num_samples*batch_size, -1) #-> [num_s * B, D]
            y = self(x, head_idx=head_idx)
        y = y.view(num_samples, batch_size, -1) # -> [num_samp, B, D]
        avg_pred = y.mean(dim=0) #-> reduce over num_samp and get [B,D]
        avg_pred_entropy = -torch.sum(avg_pred * torch.log(avg_pred + 1e-8), dim=-1)
        entropies = -torch.sum(y * torch.log(y + 1e-8), dim=-1)
        avg_entropy = torch.mean(entropies, dim=0)
        MI = avg_pred_entropy - avg_entropy
        if not return_predictions:
            return MI
        else:
            return MI, avg_pred
    
    def calc_MI(self, x):
        if self.mode != "regression":
            print("Warning, no closed from solution for softmax head!")
        pass


    def get_heads(self):
        heads = copy.deepcopy(self.heads)
        return heads
    
    def set_heads(self, heads):
        self.heads = copy.deepcopy(heads)
    
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
        device = self.heads[-1].W_mu.device if len(self.heads) > 0 else self.W_mu.device
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
    
    def forward_with_routing(self, x, mode="batchwise", num_samples=25):
        if mode != "batchwise":
            print("Please use batchwise mode, other modes not impelemented")

        MIs = []
        preds = []
        best_head = None
        for head_idx in range(len(self.heads)):
            MI, avg_pred = self.get_MI(x, num_samples=num_samples, return_predictions=True, head_idx=head_idx)
            avg_MI = MI.mean(dim=0)
            MIs.append(avg_MI)
            preds.append(avg_pred)
        MIs = torch.tensor(MIs)
        best_head = torch.argmin(MIs)
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

        if self.mode == "bernoulli":
            probs = F.softmax(y, dim=-1)
        elif self.mode=="regression":
            probs = y
        return probs
                