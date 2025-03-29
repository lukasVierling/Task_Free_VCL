import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):
        #encode
        z = self.encoder(x)
        #activation
        h = F.relu(z)
        #decode
        x_hat = self.decoder(h)
        #sigmoid?
        x_hat = torch.sigmoid(x_hat)

        return x_hat

    def mse_loss(self, x):
        ground_truth = x
        reconsturcted = self(x)
        return F.mse_loss(ground_truth, reconsturcted)
    
    def nll_loss(self, x):
        ground_truth = x
        reconstructed = self(x)
        return F.binary_cross_entropy(reconstructed, ground_truth)


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


class DiscriminativeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, autoencoder_hidden_dim):
        super().__init__()
        self.input_dim =input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.autoencoder_hidden_dim = autoencoder_hidden_dim
        self.active_head = 0

        self.encoder= BayesianLinear(input_dim, hidden_dim)

        self.heads = nn.ModuleList()
        self.autoencoders = nn.ModuleList()

    def kl_divergence(self):
        
        kl_div = self.encoder.kl_divergence()
        return kl_div

    def add_head(self):
        device = self.encoder.W_mu.device
        head = BayesianLinear(self.hidden_dim, self.output_dim).to(device)
        #add the new head
        self.heads.append(head)
        #set idx
        self.active_head = len(self.heads)-1
        #add autoencoder for head
        autoencoder = AutoEncoder(self.input_dim, self.autoencoder_hidden_dim).to(device)
        self.autoencoders.append(autoencoder)
        print(f"added new head : {self.active_head} and new autoencoder: {len(self.heads), len(self.autoencoders)}")

    def activate_head(self, head_idx):
        self.active_head = head_idx

    def get_active_head_idx(self):
        return self.active_head
    
    def forward(self, x, head_idx = None, return_reconstruction_loss="no"):
        #forward throught the model with optional reconst loss of current autoencoder 
        if head_idx is None:
            head_idx = self.active_head
        z = self.encoder(x)
        h = F.relu(z)
        logits = self.heads[head_idx](h)

        if return_reconstruction_loss == "nll":
            recon_loss = self.autoencoders[head_idx].nll_loss(x)
            return logits, recon_loss
        if return_reconstruction_loss == "mse":
            recon_loss = self.autoencoders[head_idx].mse_loss(x)
            return logits, recon_loss
        else:
            return logits
    
    def forward_with_routing(self, x, routing_mode="batchwise"):
        logits  =None
        if routing_mode != "batchwise":
            print("pelase choose a valid mode")
        else:
            if len(self.autoencoders) != len(self.heads):
                print("length of autoencoder and heads not the same please fix")
            recon_losses = [ self.autoencoders[i].mse_loss(x) for i in range(len(self.autoencoders))]
            recon_losses = torch.stack(recon_losses) 
            best_head = torch.argmin(recon_losses, dim=0).item()
            logits = self(x, head_idx=best_head, return_reconstruction_loss="no")
        return logits, best_head



