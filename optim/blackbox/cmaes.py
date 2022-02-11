import numpy as np


class CMAES:
    def __init__(self , n_samples = None):
        self.n_samples = n_samples
    def diagonal(self, d):
        x = np.eye(d.size)
        np.fill_diagonal(x, d)
        return x
    def coerce_to_search_space(self, x, search_space):
        if search_space is None:
            return x
        for i in range(x.shape[0]):
            for j,space in zip(range(x.shape[1]), search_space):
                if x[i,j] < space[0]:
                    x[i,j]=space[0]
                if x[i,j] > space[1]:
                    x[i,j]=space[1]
        return x
    def optimize(self, func, x0, search_space=None, sigma = .3, n_samples=None):
        N = x0.shape[0] # dimension of the search space
        xmean = x0   # initial point
        #sigma = .3   # coordinate wise standard deviation (step size)
        stopfitness = 1e-10 # stop is fitness < stopfitness
        stopeval = 1e3*(N**2)

        # strategy parameter setting : Selection
        if n_samples is None:
            lambd = 4+int(3*np.log(N)) # offspring number
        else:
            lambd = n_samples
        mu = int(lambd / 2)        # number of points for recombination
        weights = np.log(mu + .5) - np.log(np.arange(1, mu+1))
        weights = weights / np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)

        # strategy parameter setting : Adaptation
        cc = (4 + mueff / N) / (N+4 + 2*mueff/N) # time constant for cumulation for C
        cs = (mueff+2) / (N+mueff+5)             # time constant for cumulation for sigma control
        c1 = 2. / ((N+1.3)**2 + mueff)           # learning rate for rank-one update of C
        cmu = np.min([1.-c1, 2*(mueff-2.+1./mueff) / ((N+2)**2 + mueff)]) # lr for rank-mu update
        damps = 1. + 2*np.max([0., np.sqrt((mueff-1.)/(N+1))-1.]) + cs # damping for sigma, usually close to 1

        # Initialize dynamic strategy parameters and constants
        pc = np.zeros((N,1)) # evolution path for C
        ps = np.zeros((N,1)) # evolution path for sigma
        B = np.eye(N)      # coordinate system
        D = np.ones((N,1))   # diagonal D defines the scaling
        C = np.dot(B, np.dot(self.diagonal(D**2), B.T)) # covariance matrix C
        invsqrtC = np.dot(B, np.dot(self.diagonal(1./D), B.T)) # C^-1/2
        eigeneval = 0       # track update of B and D
        chiN = np.sqrt(N)*(1. - 1./(4*N) + 1./(21.*N**2)) # Expectation of ||N(0,I)|| == normal(randn(N,1))

        # main loop
        counteval = 0
        while counteval < stopeval:
            #print(counteval)
            # generate offspring
            #arx = np.random.multivariate_normal(xmean, sigma**2. * C, size=lambd)
            arx = np.zeros((lambd, N))
            for i in range(lambd):
                rr = np.random.normal(size=(N,1))
                temp =  xmean + (sigma * np.dot(B, D * rr)).reshape(N)
                arx[i,:] =temp # xmean + (sigma * np.dot(B, D * np.random.normal(size=(N,1)))).reshape(N)
            self.coerce_to_search_space(arx, search_space)
            arfitness = np.array([func(xi) for xi in arx])
            counteval += lambd

            # sort by fitness and compute weighted mean
            arindex = np.argsort(arfitness)
            xold = xmean
            xmean = np.zeros(xold.shape)
            for i in range(mu):
                xmean += arx[arindex[i]] * weights[i]
            #xmean = np.dot(arx[arindex[:mu],:].T, weights)

            # Cumulation : update evolution paths
            ps = (1. - cs)*ps + np.sqrt(cs*(2-cs)*mueff) * np.dot(invsqrtC, (xmean-xold) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*counteval/lambd))/chiN < 1.4 / 2/(N+1)
            pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma

            # Adapt covariance matrix C
            artmp = (1./sigma) * arx[arindex[:mu],:] - xold
            aa = np.dot(artmp.T, np.dot(self.diagonal(weights), artmp))
            C = (1-c1-cmu) * C + c1 * (np.dot(pc, pc.T) + (1-hsig)*cc*(2-cc)*C) + \
                    cmu * np.dot(artmp.T, np.dot(self.diagonal(weights), artmp))

            # Adapt the step size
            sigma = sigma * np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1.))

            # Decomposition of C into B*diag(D**2)*B.T
            if counteval - eigeneval > lambd/(c1+cmu)/N/10:
                try:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C,1)
                    D,B = np.linalg.eigh(C)
                    D = np.sqrt(self.diagonal(D))
                    invD = self.diagonal(1./np.diag(D))
                    invsqrtC = np.dot(B, np.dot(invD, B.T))
                    D = np.diag(D).reshape(N,1)
                except Exception as e:
                    print(e)
                    return xmean, arx[arindex[0]]


            # Break if fitness is good enough or condition exceeds 1e14, other termination method possible
            if (arfitness[arindex[0]] <= stopfitness) or (np.max(D) > 1e7*np.min(D)):
                if (arfitness[arindex[0]] <= stopfitness):
                    #print("Fitness achieved")
                    pass
                if (np.max(D) > 1e7*np.min(D)):
                    #print("Bad conditioned matrix")
                    pass
                break
        #print("Counteval : ", counteval, stopeval)
        return xmean, arx[arindex[0]]

if __name__=="__main__":
    x0 = np.random.random_sample(2)
    print("x0 : {}".format(x0))


    def F(x):
        return x[0]**2 + x[1]**2

    optimizer = CMAES()
    print(optimizer.optimize(F, x0, n_samples=100))
