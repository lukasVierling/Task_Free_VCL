import torch

def kl_div_gaussian(p,q):
    '''
    calculates KL(p||q) in closed form solution
    '''
    log_var_1 = p["W_sigma"]
    mu_1 = p["W_mu"]
    log_var_2 = q["W_sigma"]
    mu_2 = q["W_mu"]

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

    weights_loss = 0.5*(trace_part + squared_part - d + log_part)

    ###
    #bias
    ####

    log_var_1 = p["b_sigma"]
    mu_1 = p["b_mu"]
    log_var_2 = q["b_sigma"]
    mu_2 = q["b_mu"]

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