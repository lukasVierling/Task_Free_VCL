import torch

def kl_div_gaussian(q,p):
    '''
    calculates KL(q||p) in closed form solution
    '''
    Sigma_1 = q["W_sigma"]
    mu_1 = q["W_mu"]
    Sigma_2 = p["W_sigma"]
    mu_2 = p["W_mu"]

    #flatten into vectors
    Sigma_1 = Sigma_1.view(-1)
    mu_1 = mu_1.view(-1)
    Sigma_2 = Sigma_2.view(-1)
    mu_2 = mu_2.view(-1)


    #calc the KL divergence
    trace_part = torch.sum(Sigma_2 / Sigma_1) #use that matrices are diagonal so we can just divide
    squared_part = ((mu_2-mu_1)/Sigma_1).T @ (mu_2 -mu_1)
    d = Sigma_1.shape[0] #dimension is the lenght for squared matrices
    log_part = torch.sum(torch.log(Sigma_1)) - torch.sum(torch.log(Sigma_2))
    return 0.5*(trace_part + squared_part - d + log_part)