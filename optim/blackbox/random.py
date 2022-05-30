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


