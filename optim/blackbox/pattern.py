import numpy as np

class NelderMead:
    def __init__(self, alpha=1., gamma=2., rho=.5, sigma=.5, tolerance=1e-4):
        if alpha <= 0.:
            raise Exception("NelderMead alpha parameter should be strictly greater than 0, got {}".format(alpha))

        self.alpha = alpha
        if gamma <= 1.:
            raise Exception("NelderMead gamma parameter should be strictly greater than 1, got {}".format(gamma))
        self.gamma = gamma
        if rho<=0. or rho>.5:
            raise Exception("NelderMead rho paramter must be in ]0,.5], got {}".format(rho))
        self.rho = rho
        
        self.sigma = sigma

        self.tolerance = tolerance

        self.it=0
        self.history=None
    def optimize(self, func, x0):
        simplex = x0
        n = x0.shape[0]
        if len(x0.shape)!=2:
            simplex = x0 + np.random.random_sample((n+1,n))
        else:
            simplex = x0
        self.it=0
        self.history={"simplex":[], "evaluation":[]}
        while self.it!=1000:
            if len(self.history["simplex"])>2:
                #print(self.it, np.all(self.history["simplex"][-2]==simplex))
                if np.all(self.history["simplex"][-2]==simplex):
                    break
                if np.all(np.linalg.norm(self.history["simplex"][-2]-simplex, axis=1) < self.tolerance):
                    break
            self.history["simplex"].append(np.copy(simplex))
            # evaluate and order the simplex vertices
            simplex_evaluation = np.fromiter((func(xi) for xi in simplex),dtype=simplex.dtype)
            ordered_evaluation = np.sort(simplex_evaluation)
            ordered_simplex = simplex[np.argsort(simplex_evaluation)]
            # centroid
            centroid = np.mean(ordered_simplex[:-1], axis=0)
            # reflection
            reflection = centroid + self.alpha * (centroid - ordered_simplex[-1])
            reflection_evaluation = func(reflection)
            if reflection_evaluation < ordered_evaluation[-2] and reflection_evaluation>=ordered_evaluation[0]:
                simplex[:-1] = ordered_simplex[:-1]
                simplex[-1] = reflection
                self.it+=1
                #self.history["simplex"].append(simplex)
                continue
            # expansion
            if reflection_evaluation<ordered_evaluation[0]:
                # expanded point
                expansion = centroid + self.gamma * (reflection - centroid)
                expansion_evaluation = func(expansion)
                simplex[:-1] = ordered_simplex[:-1]
                if expansion_evaluation < reflection_evaluation:
                    simplex[-1] = expansion
                else:
                    simplex[-1] = reflection
                self.it+=1
                #self.history["simplex"].append(simplex)
                continue
            # contraction
            contraction = centroid + self.rho * (ordered_simplex[-1] - centroid)
            contraction_evaluation = func(contraction)
            if contraction_evaluation < ordered_evaluation[-1]:
                simplex[:-1] = ordered_simplex[:-1]
                simplex[-1] = contraction
                self.it+=1
                #self.history["simplex"].append(simplex)
                continue
            # shrink
            ordered_simplex[1:] = ordered_simplex[0] + self.sigma * (ordered_simplex[1:] - ordered_simplex[0])
            simplex = ordered_simplex
            self.it+=1
            #self.history["simplex"].append(simplex)

        simplex_evaluation = np.fromiter((func(xi) for xi in simplex),dtype=simplex.dtype)
        ordered_simplex = simplex[np.argsort(simplex_evaluation)]
        return ordered_simplex[0]


