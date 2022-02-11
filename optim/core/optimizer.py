"""Optimizer class definition : Base classed shared by all optimizer objects. Exposes the optimize function that solves the optimization problem for a given function.

An optimizer has a set of parameters. Most of the optimizers are iterative processes and will have a 
"max_iter" attribute. 

The optimize function returns a History object. The current state of the problem is stored in the
Context.
"""

from abc import ABC

from optim.utils import dict_sum

class Optimizer(ABC):
    def __init__(self, max_iter = 1000, **kwargs):
        # parameters
        self.max_iter = max_iter
        self.__dict__.update(kwargs)
    #def terminate_context(self, context)
    def termination_criterion(self, context):
        # check the context for termination
        return context["iteration"] == self.max_iter
    def update_context(self, context):
        # increment the iteration counter
        context["iteration"]+=1
    def create_context(self, function, x0, **kwargs):
        # initialize the iteration counter
        context = {"iteration":0, "function":function, "x0":x0}
        return context
    def optimize(self, function, x0, **kwargs):
        # create a new optimization context for the function and arguments that were passed
        context = self.create_context(function, x0, **kwargs)
        # clear history
        history = []
        while not self.termination_criterion(context):
            # modify the context
            self.update_context(context)
            # update history
            history.append(context.copy())
        # return the final context
        return history
