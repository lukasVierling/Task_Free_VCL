import torch
import torch.nn as nn
import torch.nn.functional as F

def get_standard_normal_prior_extension(model, device):
        input_dim = model.input_dim
        hidden_dim = model.hidden_dim
        output_dim = model.output_dim
        single_head = model.single_head
        encoder = {
            "W_mu": nn.Parameter(torch.zeros(input_dim, hidden_dim)).to(device),
            "b_mu": nn.Parameter(torch.zeros(hidden_dim)).to(device),
            "W_sigma": nn.Parameter(torch.zeros(input_dim, hidden_dim)).to(device),
            "b_sigma": nn.Parameter(torch.zeros(hidden_dim)).to(device)
        }
        var_dist = {}
        var_dist["encoder"] = encoder

        
        return var_dist["encoder"]

def get_mle_estimate_extension(model, dataset, device):
    input_dim = model.input_dim
    hidden_dim = model.hidden_dim
    output_dim = model.output_dim
    mle_model = nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))
    optimizer = torch.optim.Adam(mle_model.parameters(), lr=0.001)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=256)
    #get MLE estimate after tarining for a single epoch (should be enough on MNIST)
    mle_model.to(device)
    for x,y in data_loader:
        mle_model.zero_grad()
        x,y = x.to(device),y.to(device)
        x = x.view(x.shape[0], -1)
        outputs = mle_model(x)
        loss = 0
        if model.mode == "regression":
            print("no regression MLE support yet")
            one_hot_labels = F.one_hot(y, num_classes=model.output_dim).float()
            loss = F.mse_loss(outputs,one_hot_labels)

        elif model.mode == "bernoulli":
            #outputs are not softmaxed
            loss = F.cross_entropy(outputs, y)
        else:
            print("Choose supported mode for MLE estimate")

        loss.backward()
        optimizer.step()
    #finished training now save the MLE 
    var_dist = {}
    encoder = {
        "W_mu":  nn.Parameter(mle_model[0].weight.data.detach().clone().t()), #transpose because nn.Linear(M,N) creates NxM matrix ! 
        "b_mu":  nn.Parameter(mle_model[0].bias.data.detach().clone()),
        "W_sigma": nn.Parameter(torch.full(mle_model[0].weight.data.shape, -6.0).t()),
        "b_sigma": nn.Parameter(torch.full(mle_model[0].bias.data.shape, -6.0))
    }
    var_dist["encoder"] = encoder

    var_dist["heads"] = [{
        "W_mu":  nn.Parameter(mle_model[2].weight.data.detach().clone().t()),
        "b_mu":  nn.Parameter(mle_model[2].bias.data.detach().clone()),
        "W_sigma": nn.Parameter(torch.full(mle_model[2].weight.data.shape, -6.0).t()),
        "b_sigma": nn.Parameter(torch.full(mle_model[2].bias.data.shape, -6.0))
    }]
    return var_dist



def get_standard_normal_prior(model, device):
        input_dim = model.input_dim
        hidden_dim = model.hidden_dim
        output_dim = model.output_dim
        single_head = model.single_head
        encoder_1 = {
            "W_mu": nn.Parameter(torch.zeros(input_dim, hidden_dim)).to(device),
            "b_mu": nn.Parameter(torch.zeros(hidden_dim)).to(device),
            "W_sigma": nn.Parameter(torch.zeros(input_dim, hidden_dim)).to(device),
            "b_sigma": nn.Parameter(torch.zeros(hidden_dim)).to(device)
        }
        encoder_2 = {
            "W_mu": nn.Parameter(torch.zeros(hidden_dim, hidden_dim)).to(device),
            "b_mu": nn.Parameter(torch.zeros(hidden_dim)).to(device),
            "W_sigma": nn.Parameter(torch.zeros(hidden_dim, hidden_dim)).to(device),
            "b_sigma": nn.Parameter(torch.zeros(hidden_dim)).to(device)
        }
        var_dist = {}
        var_dist["encoder1"] = encoder_1
        var_dist["encoder2"] = encoder_2

        if single_head:
            var_dist["heads"] = [{
                "W_mu": nn.Parameter(torch.zeros(hidden_dim, output_dim)).to(device),
                "b_mu": nn.Parameter(torch.zeros(output_dim)).to(device),
                "W_sigma": nn.Parameter(torch.zeros(hidden_dim, output_dim)).to(device),
                "b_sigma": nn.Parameter(torch.zeros(output_dim)).to(device)
            }]

            return var_dist

def get_mle_estimate(model, dataset, device):
    input_dim = model.input_dim
    hidden_dim = model.hidden_dim
    output_dim = model.output_dim
    mle_model = nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))
    optimizer = torch.optim.Adam(mle_model.parameters(), lr=0.001)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=256)
    #get MLE estimate after tarining for a single epoch (should be enough on MNIST)
    mle_model.to(device)
    for x,y in data_loader:
        mle_model.zero_grad()
        x,y = x.to(device),y.to(device)
        x = x.view(x.shape[0], -1)
        outputs = mle_model(x)
        loss = 0
        if model.mode == "regression":
            print("no regression MLE support yet")
            one_hot_labels = F.one_hot(y, num_classes=model.output_dim).float()
            loss = F.mse_loss(outputs,one_hot_labels)

        elif model.mode == "bernoulli":
            #outputs are not softmaxed
            loss = F.cross_entropy(outputs, y)
        else:
            print("Choose supported mode for MLE estimate")

        loss.backward()
        optimizer.step()
    #finished training now save the MLE 
    var_dist = {}
    encoder1 = {
        "W_mu":  nn.Parameter(mle_model[0].weight.data.detach().clone().t()), #transpose because nn.Linear(M,N) creates NxM matrix ! 
        "b_mu":  nn.Parameter(mle_model[0].bias.data.detach().clone()),
        "W_sigma": nn.Parameter(torch.full(mle_model[0].weight.data.shape, -6.0).t()),
        "b_sigma": nn.Parameter(torch.full(mle_model[0].bias.data.shape, -6.0))
    }
    var_dist["encoder1"] = encoder1
    encoder2 = {
        "W_mu":  nn.Parameter(mle_model[2].weight.data.detach().clone().t()), #transpose because nn.Linear(M,N) creates NxM matrix ! 
        "b_mu":  nn.Parameter(mle_model[2].bias.data.detach().clone()),
        "W_sigma": nn.Parameter(torch.full(mle_model[2].weight.data.shape, -6.0).t()),
        "b_sigma": nn.Parameter(torch.full(mle_model[2].bias.data.shape, -6.0))
    }
    var_dist["encoder2"] = encoder2

    if model.single_head:
        var_dist["heads"] = [{
            "W_mu":  nn.Parameter(mle_model[4].weight.data.detach().clone().t()),
            "b_mu":  nn.Parameter(mle_model[4].bias.data.detach().clone()),
            "W_sigma": nn.Parameter(torch.full(mle_model[4].weight.data.shape, -6.0).t()),
            "b_sigma": nn.Parameter(torch.full(mle_model[4].bias.data.shape, -6.0))
        }]
    return var_dist

def kl_div_gaussian(p,q):
    loss = 0
    #encoder 1
    p_encoder_1 = p["encoder1"]
    q_encoder_1 = q["encoder1"]
    loss += kl_div_gaussian_layer(p_encoder_1,q_encoder_1)
    # encoder 2
    p_encoder_2 = p["encoder2"]
    q_encoder_2 = q["encoder2"]
    loss += kl_div_gaussian_layer(p_encoder_2,q_encoder_2)
    #should usually be only 1 head for single head setup

    p_heads = p["heads"]
    q_heads = q["heads"]
    if len(q_heads) > 1:
        print("Warning, calc kl div over more than one head!")
    for p,q in zip(p_heads, q_heads):
        #print("calc kl loss over head")
        loss += kl_div_gaussian_layer(p,q)
    return loss


def kl_div_gaussian_layer(p,q):
    '''
    calculates KL(p||q) in closed form solution
    '''
    log_var_1 = p["W_sigma"]
    mu_1 = p["W_mu"]
    log_var_2 = q["W_sigma"]
    mu_2 = q["W_mu"]

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