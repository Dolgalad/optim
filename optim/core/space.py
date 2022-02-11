"""Definition of the Space object.
"""
import numpy as np

class Space:
    def __init__(self):
        pass
    def __contains__(self, x):
        return False
    def __eq__(self, x):
        return False

class Interval(Space):
    def __init__(self, bounds=[0,1], closed=[True,True]):
        self.bounds = bounds
        self.closed = closed
    def __contains__(self, x):
        if isinstance(x, int) or isinstance(x, float):
            return x>=self.bounds[0] and x<=self.bounds[1]
        if isinstance(x, Interval):
            return x.bounds[0]>=self.bounds[0] and x.bounds[1]<=self.bounds[1]
        return False
    def __eq__(self, x):
        return x.bounds[0]==self.bounds[0] and x.bounds[1]==self.bounds[1]
    def linspace(self, n):
        return np.linspace(self.bounds[0], self.bounds[1], n)

class CartesianProduct(Space):
    def __init__(self, spaces=[]):
        self.spaces = spaces
    def __contains__(self, x):
        if isinstance(x, np.array):
            if x.shape[0]==len(self.spaces):
                for xi, space in zip(x, self.spaces):
                    if not xi in space:
                        return False
        return False

class IntervalProduct(CartesianProduct):
    def __init__(self, spaces=[]):
        for i in range(len(spaces)):
            if isinstance(spaces[i], list):
                spaces[i]=Interval(bounds=spaces[i])
        super().__init__(spaces)
    def meshgrid(self, n=100):
        if len(self.spaces)==1:
            return self.spaces[0].linspace(n)
        return np.meshgrid(*(space.linspace(n) for space in self.spaces))
