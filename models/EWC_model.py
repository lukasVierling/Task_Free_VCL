import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, Subset



class DiscriminativeModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # first hidden layer -> fld layer in one parameter vector
        # Bayesian first hidden layer parameters (mean & log variance)
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        # second hidden layer, bayesian layer
        self.heads = nn.ModuleList()
        self.active_head = 0

    def get_stacked_params(self, detach=True):
        if detach:
            params = [self.linear.weight.clone().detach().view(-1),
                self.linear.bias.clone().detach().view(-1)]
        else:
            params = [self.linear.weight.view(-1),
                self.linear.bias.view(-1)]
        params = torch.cat(params)
        return params

    def get_fisher(self, dataset,sample_size=5000):
        fisher_diag = None
        self.eval()
        batch_size = 1 #to prevent weird errors from summing before squaring
        device = self.linear.weight.device
        idx = torch.randperm(len(dataset))[:sample_size]
        subset = Subset(dataset, idx)
        data_loader = DataLoader(subset, shuffle=False, batch_size=batch_size) #shuffle doesn't matter 
        #calculate the fisher information matrix on dataset D for current parameters
        for x,y in data_loader:

            self.zero_grad()
            x,y = x.to(device), y.to(device)
            output = self(x)
            probs = output.gather(1, y.view(-1, 1)).squeeze() # get p(y_t | theta, x_t)
            log_probs = torch.log(probs + 1e-8) #calc log(p(..))
            loss = -log_probs.mean() #Sign shouldn't matter
            #take the gradient
            loss.backward()
            # concat and flatten all the gradients
            grads = torch.cat([self.linear.weight.grad.clone().detach().view(-1),
                                 self.linear.bias.grad.clone().detach().view(-1)])
            #square the gradient
            squared_grads = grads ** 2
            if fisher_diag is None:
                fisher_diag = torch.zeros_like(squared_grads, device=device)
            fisher_diag += squared_grads
            
            #TODO paper doesn't average but Probabilistic ML 2 in formula 3.53 averages
        self.zero_grad()
        fisher_diag = fisher_diag/sample_size #TODO consider if we should take mean or sum
        self.train()
        return fisher_diag

    def get_heads(self):
        heads = copy.deepcopy(self.heads)
        return heads
    
    def set_heads(self, heads):
        self.heads = copy.deepcopy(heads)

    
    def add_head(self):
        '''
        add a new head to the model and set active head to this head
        '''
        # move the old head to the same device as previous head
        device = self.heads[-1].weight.device if len(self.heads) > 0 else self.linear.weight.device
        new_head = nn.Linear(self.hidden_dim, self.output_dim).to(device)
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

        z = F.relu(self.linear(x))
        
        # forward throught head
        y = self.heads[self.active_head](z)
        probs = F.softmax(y, dim=-1)
        return probs
