class GaussianMeanField():
    '''
    Class for variational disribution, the weights are stacked to one vector, we assume q_t(theta) = Pord N(theta, mu, sigma^2), cov matrix is diagonal
    '''
    def __init__(self, mus, sigmas):
        self.mus = mus #TODO make this parameters
        self.sigmas = sigmas #TODO make this parameters


