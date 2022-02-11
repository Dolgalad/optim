"""Optimization problem context
"""

from abc import ABC

class Context(ABC):
    def __init__(self, **kwargs):
        self.max_iter = 1000
        if "max_iter" in kwargs:
            self.max_iter = kwargs["max_iter"]
