"""
Linear Program implementation
"""

import numpy as np

from optim.tex_utils import *
from optim.problems import Problem

class LinearProgram(Problem):
    def __init__(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        """
        Linear program defined in its canonical form :
        min c.T x
        st. A x <= b
            x >= 0
        """
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds
        

    def canonical_form(self):
        """Return problem in canonical form
        """
        # upper bound constrain matrix
        new_A_ub = self.A_ub
        new_b_ub = self.b_ub
        if self.A_eq is not None:
            # add equality constraint as an upper and lower bound
            new_A_ub = np.vstack((new_A_ub, self.A_eq, -self.A_eq))
            new_b_ub = np.hstack((new_b_ub, self.b_eq, -self.b_eq))
        if self.bounds is not None:
            if self.bounds[0] is not None:
                # lower bound
                new_A_ub = np.vstack((new_A_ub, -np.eye(self.c.shape[0])))
                new_b_ub = np.hstack((new_b_ub, -self.bounds[0]))
            if self.bounds[1] is not None:
                # upper bound
                new_A_ub = np.vstack((new_A_ub, np.eye(self.c.shape[0])))
                new_b_ub = np.hstack((new_b_ub, self.bounds[1]))
        return LinearProgram(self.c, new_A_ub, new_b_ub)

    def augmented_form(self):
        """Get augmented form of the problem, introduces additional slack variables
        """
        nc = np.zeros(1 + 2 * self.A_ub.shape[0])
        nc[0] = 1
        Ablocks = []
        bblocks = []
        if self.A_ub is not None:
            Ablocks.append([np.block([[np.ones(1), -self.c.T, np.zeros(self.A_ub.shape[0])],
                    [np.zeros((self.A_ub.shape[0],1)), self.A_ub, np.eye(self.A_ub.shape[0])]])])
            bblocks += [np.zeros(1), self.b_ub]
        if self.A_eq is not None:
            Ablocks.append([np.block([np.zeros((self.A_eq.shape[0],1)), self.A_eq, np.zeros((self.A_eq.shape[0],self.A_eq.shape[0]))])])
            bblocks += [self.b_eq]
        nAeq = np.block(Ablocks)
        nbeq = np.block(bblocks)
        return LinearProgram(nc, 
                A_eq=nAeq,
                b_eq=nbeq
                )


    def _tex_repr_(self):
        """LaTex representation
        """
        x = (np.arange(self.c.shape[0])+1).astype(int)
        x = np.array([r"x_{" + str(xi) + r"}" for xi in x])
        ans = r"\begin{array}{cl}"
        ans += r"\min\limits_{x \in \mathbb{R}^" + str(self.c.shape[0]) + r"} & " + array_to_tex(self.c,vertical=False) + r" " + array_to_tex(x) + r" \\"
        if self.A_ub is not None:
            ans += r"\mbox{st.} & " + array_to_tex(self.A_ub) + r" " + array_to_tex(x) + r"\leq " + array_to_tex(self.b_ub) + r"\\"
        if self.A_eq is not None:
            ans += r" & " + array_to_tex(self.A_eq) + r" " + array_to_tex(x) + r" = " + array_to_tex(self.b_eq) + r"\\"
        if self.bounds is not None:
            if self.bounds[0] is None or self.bounds[1] is None:
                if self.bounds[0] is not None:
                    ans += r" & " + array_to_tex(x) + " \geq" + array_to_tex(self.bounds[0])
                if self.bounds[1] is not None:
                    ans += r" & " + array_to_tex(x) + r" \leq " + array_to_tex(self.bounds[1])
            else:
                ans += r" & " + array_to_tex(self.bounds[0]) + r" \leq " + array_to_tex(x) + r" \leq " + array_to_tex(self.bounds[1])
        else:
            ans += r" & x \geq 0"

        ans += r"\end{array}"
        return ans

    def scipy_bounds(self):
        if self.bounds is None:
            return self.c.shape[0] * [(0, None)]
        else:
            pass


if __name__=="__main__":
    lp = LinearProgram(np.ones(2), np.eye(2), 2*np.ones(2))





