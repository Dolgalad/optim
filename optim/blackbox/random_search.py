"""Random search optimizer
"""
import numpy as np

from optim.utils import dict_sum
from optim.core.optimizer import Optimizer

class RandomSearch(Optimizer):
    def __init__(self, n_samples = 1, **kwargs):
        c = dict_sum({"n_samples":n_samples}, kwargs)
        super().__init__(**c)
    def create_context(self, function, x0, sampling_method=None, search_space=None, **kwargs):
        # default context 
        context = super().create_context(function, x0, **kwargs)
        return context
    def draw_samples(self, x0):
        if self.sampling_method is None:
            sm = lambda x : self.context["x0"] + 2.*(np.random.random_sample(size=(self.n_samples,1)))

        # the default random search will draw a single sample around the current point x0
        if len(self.context["x0"].shape)==0:
            x = self.context["x0"] + 2.*(np.random.random_sample(size=(self.context["n_samples"],1)))
        else:
            x = self.context["x0"] + 2.*(np.random.random_sample(size=[self.context["n_samples"]]+
                list(self.context["x0"])))
            return x

    def update_context(self):
        # update the context
        # the default random search will draw a single sample around the current point x0
        if len(self.context["x0"].shape)==0:
            x = self.context["x0"] + 2.*(np.random.random_sample(size=(self.context["n_samples"],1)))
        else:
            x = self.context["x0"] + 2.*(np.random.random_sample(size=[self.context["n_samples"]]+
                list(self.context["x0"])))

        new_point = self.context["sampling_method"](self.context["x0"])

        # evaluate the current point
        if self.context["n_samples"]!=1:
            new_eval = np.array([self.context["function"](xr) for xr in x])
        else:
            new_eval = self.context["function"](x)
        prev_eval = self.context["function"](self.context["x0"])
        if new_eval < prev_eval:
            self.context["x0"]=new_point
        # increment the iteration counter
        super().update_context()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    print("Test the RandomSeach optimizer object")
    def F1(x):
        return x**2

    # fix the random generator seed
    np.random.seed(1024)

    optimizer = RandomSearch(max_iter = 10)
    history = optimizer.optimize(F1, 5., max_iter=10)
    print("Default RandomSearch")
    print("\tLast context   : {}".format(history[-1]))
    print("\tHistory length : {}".format(len(history)))
    solution = history[-1]
    print("\tFinal point    : {}".format(solution["x0"]))
    print("\tFinal eval     : {}".format(F1(solution["x0"])))
    print()

    # try a different sampling method
    def sampling_test1(x0):
        return x0 + np.random.normal()
    optimizer = RandomSearch(max_iter = 10, sampling_method=sampling_test1)
    #history = optimizer.optimize(F1, 5., max_iter=10, sampling_method=sampling_test1)
    history = optimizer.optimize(F1, 5.)

    print("Default RandomSearch with custom sampling")
    print("\tLast context   : {}".format(history[-1]))
    print("\tHistory length : {}".format(len(history)))
    solution = history[-1]
    print("\tFinal point    : {}".format(solution["x0"]))
    print("\tFinal eval     : {}".format(F1(solution["x0"])))
    print()

    # define a two dimensional function
    def F2(x):
        return x[0]**2 + x[1]**2
    print("Default RandomSearch with two dimension function")
    history = optimizer.optimize(F2, [5., 10.], max_iter=10)
    print("\tLast context   : {}".format(history[-1]))
    print("\tHistory length : {}".format(len(history)))
    solution = history[-1]
    print("\tFinal point    : {}".format(solution["x0"]))
    print("\tFinal eval     : {}".format(F2(solution["x0"])))
    print()

    print("Default RandomSearch with two dimension function and sampling multiple points")
    def multi_sampling(x0):
        return x0 + np.random.normal()
    optimizer = RandomSearch(n_samples=10, max_iter=10)
    history = optimizer.optimize(F2, [5., 10.])
    print("\tLast context   : {}".format(history[-1]))
    print("\tHistory length : {}".format(len(history)))
    solution = history[-1]
    print("\tFinal point    : {}".format(solution["x0"]))
    print("\tFinal eval     : {}".format(F2(solution["x0"])))
    print()




