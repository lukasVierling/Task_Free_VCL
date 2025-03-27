import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm


class DiscriminativeModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, single_head=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # first hidden layer -> fld layer in one parameter vector
        # Bayesian first hidden layer parameters (mean & log variance)
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # second hidden layer, bayesian layer
        self.heads = nn.ModuleList()
        self.active_head = 0
        self.single_head = single_head
        if self.single_head:
            print("using single head")
            self.add_head()

    def get_stacked_params(self, detach=True):
        if detach:
            params = [
                self.linear.weight.clone().detach().view(-1),
                self.linear.bias.clone().detach().view(-1)
                ]
            params2 = [
                self.linear2.weight.clone().detach().view(-1),
                self.linear2.bias.clone().detach().view(-1)
                ]
            params = params + params2

            if self.single_head:
                head = self.heads[-1]
                head_params = [
                        head.weight.clone().detach().view(-1),
                        head.bias.clone().detach().view(-1)
                        ]
                #concat the heads
                params = params + head_params
        else:
            params = [self.linear.weight.view(-1),
                self.linear.bias.view(-1)]
            params2 = [
                self.linear2.weight.view(-1),
                self.linear2.bias.view(-1)
                ]
            params = params + params2
            
            if self.single_head:
                head = self.heads[-1]
                head_params = [
                        head.weight.view(-1),
                        head.bias.view(-1)
                ]
                #concat the heads
                params = params + head_params

        params = torch.cat(params)
        return params

    def get_fisher(self, dataset,sample_size=600):
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
            loss = log_probs.mean() #Sign shouldn't matter
            #take the gradient
            loss.backward()
            # concat and flatten all the gradients
            grads = torch.cat([self.linear.weight.grad.clone().detach().view(-1),
                                 self.linear.bias.grad.clone().detach().view(-1)])
            grads2 = torch.cat([self.linear2.weight.grad.clone().detach().view(-1),
                                 self.linear2.bias.grad.clone().detach().view(-1)])
            grads = torch.cat([grads,grads2])
            if self.single_head:
                #should only be a single head there
                head = self.heads[-1]
                head_grads = torch.cat([head.weight.grad.clone().detach().view(-1),
                                 head.bias.grad.clone().detach().view(-1)])
                grads = torch.cat([grads, head_grads])
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
    

    def get_hessian(self, dataset, subset_size = 500):
        print("Not used anymore")
        hessian_diag = None
        self.eval()
        batch_size = subset_size
        device = self.linear.weight.device
        idx = torch.randperm(len(dataset))[:subset_size]
        sub_dataset = Subset(dataset, idx)
        data_loader = DataLoader(sub_dataset, shuffle=True, batch_size=batch_size)
        #calculate the fisher information matrix on dataset D for current parameters
        for x,y in tqdm(data_loader, desc="Calc the Hessian"):
            x,y = x.to(device), y.to(device)
            output = self(x)
            probs = output.gather(1, y.view(-1, 1)).squeeze() # get p(y_t | theta, x_t)
            log_probs = torch.log(probs + 1e-8) #calc log(p(..))
            loss = -log_probs.sum() # SUM (log(p(..))) #convert into negative log likelihood
            #take the gradient
            #loss.backward()
            #take the derivative twice
            # concat and flatten all the gradients
            w, b = self.linear.weight, self.linear.bias
            grads = torch.autograd.grad(loss, [w,b], create_graph=True)
            grad_w, grad_b = grads

            grad_w_flat, grad_b_flat = grad_w.view(-1), grad_b.view(-1)
            w_flat, b_flat = w.view(-1), b.view(-1)

            weight_hessian_diag = torch.zeros_like(w_flat)
            bias_hessian_diag = torch.zeros_like(b_flat)
            #calc the second derivative
            for i in range(w.numel()):
                grad_2 = torch.autograd.grad(grad_w_flat[i], w, retain_graph=True)[0].view(-1)[i]
                weight_hessian_diag[i] = grad_2
            for i in range(b.numel()):
                grad_2 = torch.autograd.grad(grad_b_flat[i], b, retain_graph=True)[0].view(-1)[i]
                bias_hessian_diag[i] = grad_2
            
            intermediate_hessian_diag = torch.cat([weight_hessian_diag, bias_hessian_diag])           
            
            if hessian_diag is None:
                hessian_diag = torch.zeros_like(intermediate_hessian_diag, device=device)
            hessian_diag += intermediate_hessian_diag
            
            #TODO paper doesn't average but Probabilistic ML 2 in formula 3.53 averages
            self.zero_grad()
        hessian_diag = hessian_diag/subset_size #TODO consider if we should take mean or sum
        self.train()
        return hessian_diag

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
        z = F.relu(self.linear2(z))
        
        # forward throught head
        y = self.heads[self.active_head](z)
        probs = F.softmax(y, dim=-1)
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
        self.linear = torch.nn.Linear(hidden_dim,output_dim)
        #Individual Encoder for every task
        self.encoders = nn.ModuleList()
        self.active_encoder = 0
        
        #Inidividual Heads
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
    

    def vae_loss(self, recon_x, x, mean, log_var):
        #eps = 1e-6 #for stability
        #likelihood part
        bernoulli_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        #now the KL part
        kl = - 0.5 * torch.sum(1+log_var - mean**2 - torch.exp(log_var))
        return bernoulli_loss + kl #both terms are positive but then flip sign later
    

    def get_fisher(self, dataset,sample_size=50000):
        fisher_diag = None
        self.eval()
        batch_size = 1 #to prevent weird errors from summing before squaring
        device = self.linear.weight.device
        idx = torch.randperm(len(dataset))[:sample_size]
        subset = Subset(dataset, idx)
        data_loader = DataLoader(subset, shuffle=False, batch_size=batch_size) #shuffle doesn't matter 
        #calculate the fisher information matrix on dataset D for current parameters
        for x,y in data_loader:
            x,y = x.to(device),y.to(device)
            self.zero_grad()
            #forward through the model to get likelihood
            output, mean, log_var = self(x) #-> returns [B,C]
            #calc the VAE loss
            
            loss = self.vae_loss(output, x, mean, log_var) #this is mean!
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

    def get_hessian(self, dataset, subset_size = 5000):
        hessian_diag = None
        self.eval()
        batch_size = subset_size
        device = self.linear.weight.device
        idx = torch.randperm(len(dataset))[:subset_size]
        sub_dataset = Subset(dataset, idx)
        data_loader = DataLoader(sub_dataset, shuffle=True, batch_size=batch_size)
        #calculate the fisher information matrix on dataset D for current parameters
        for x,y in tqdm(data_loader, desc="Calc the Hessian"):
            x,y = x.to(device),y.to(device)
            self.zero_grad()
            #forward through the model to get likelihood
            output, mean, log_var = self(x) #-> returns [B,C]
            #calc the VAE loss
            loss = self.vae_loss(output, x, mean, log_var) #this is mean! #TODO neg or pos?
            #take the gradient
            #loss.backward()
            #take the derivative twice
            # concat and flatten all the gradients
            w, b = self.linear.weight, self.linear.bias
            grads = torch.autograd.grad(loss, [w,b], create_graph=True)
            grad_w, grad_b = grads

            grad_w_flat, grad_b_flat = grad_w.view(-1), grad_b.view(-1)
            w_flat, b_flat = w.view(-1), b.view(-1)

            weight_hessian_diag = torch.zeros_like(w_flat)
            bias_hessian_diag = torch.zeros_like(b_flat)
            #calc the second derivative
            for i in range(w.numel()):
                grad_2 = torch.autograd.grad(grad_w_flat[i], w, retain_graph=True)[0].view(-1)[i]
                weight_hessian_diag[i] = grad_2
            for i in range(b.numel()):
                grad_2 = torch.autograd.grad(grad_b_flat[i], b, retain_graph=True)[0].view(-1)[i]
                bias_hessian_diag[i] = grad_2
            
            intermediate_hessian_diag = torch.cat([weight_hessian_diag, bias_hessian_diag])           
            
            if hessian_diag is None:
                hessian_diag = torch.zeros_like(intermediate_hessian_diag, device=device)
            hessian_diag += intermediate_hessian_diag
            
            #TODO paper doesn't average but Probabilistic ML 2 in formula 3.53 averages
            self.zero_grad()
        hessian_diag = hessian_diag/subset_size #TODO consider if we should take mean or sum
        self.train()
        return hessian_diag
    
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
    
    def add_head(self):
        '''
        add a new head to the model and set active head to this head
        '''
        # move the old head to the same device as previous head
        device = self.heads[-1].weight.device if len(self.heads) > 0 else self.linear.weight.device
        new_head = nn.Linear(self.latent_dim, self.hidden_dim).to(device)
        self.heads.append(new_head)
        self.active_head = len(self.heads)-1
        print("Added new head, current head index: ", self.active_head)

    def add_encoder(self):
        device = self.encoders[-1][0].weight.device if len(self.encoders) > 0 else self.linear.weight.device
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

        normalized_y = F.sigmoid(self.linear(h))

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

        normalized_y = F.sigmoid(self.linear(h))

        #normalized_y = F.sigmoid(self.last_layer(h))

        return normalized_y, mean, log_var
    

