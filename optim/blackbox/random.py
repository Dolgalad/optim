"""Black box optimization with random process
"""
import numpy as np

class RandomOptimizer:
    def __init__(self, max_iter=1000, n_samples=9, disturbance=np.random.normal, search_space=None, termination=None, early_stopping=None):
        self.early_stopping = early_stopping
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.disturbance = disturbance
        self.it = 0
        self.history = None
    def max_iter_reached(self):
        return self.it==self.max_iter
    def no_improvement(self):
        if self.history is None:
            return False
        if len(self.history["candidate"])<2:
            return False
        return np.all(self.history["candidate"][-1]==self.history["candidate"][-2])
    def has_terminated(self):
        if self.early_stopping:
            return self.max_iter_reached() or self.no_improvement()
        return self.max_iter_reached()
    def optimize(self, func, x0, verbose=False):
        self.it = 0
        self.history = {"candidate":[], "evaluation":[]}
        while not self.has_terminated():
            # draw new candidates by adding disturbance to current best candidate
            x = x0 + self.disturbance(size = (self.n_samples, x0.shape[0]))
            x = np.vstack((x0, x))
            # evaluate the function for each sample
            ev = np.fromiter([func(xi) for xi in x], dtype=x.dtype)
            # keep the best candidate
            x0 = x[np.argmin(ev)]
            if verbose:
                print(i, x0, ev[np.argmin(ev)])
            # update history
            self.history["candidate"].append(x0)
            self.history["evaluation"].append(np.min(ev))
            # increment iteration count
            self.it += 1
        return x0

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
    def optimize(self, func, simplex):
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


