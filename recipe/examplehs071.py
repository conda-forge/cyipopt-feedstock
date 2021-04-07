# -*- coding: utf-8 -*-
"""
cyipot: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://http://code.google.com/p/cyipopt/>
License: EPL 1.0
"""
#
# Test the "ipopt" Python interface on the Hock & Schittkowski test problem
# #71. See: Willi Hock and Klaus Schittkowski. (1981) Test Examples for
# Nonlinear Programming Codes. Lecture Notes in Economics and Mathematical
# Systems Vol. 187, Springer-Verlag.
#
# Based on matlab code by Peter Carbonetto.
#

import numpy as np
import scipy.sparse as sps
import cyipopt


class HS071(object):

    def __init__(self):
        self.hs = sps.coo_matrix(np.tril(np.ones((4, 4))))

    def objective(self, x):
        return x[0]*x[3]*np.sum(x[0:3]) + x[2]

    def gradient(self, x):
        return np.array([
                    x[0]*x[3] + x[3]*np.sum(x[0:3]),
                    x[0]*x[3],
                    x[0]*x[3] + 1.0,
                    x[0]*np.sum(x[0:3])
                    ])

    def constraints(self, x):
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        return np.concatenate((np.prod(x) / x, 2*x))

    def hessianstructure(self):
        return (self.hs.col, self.hs.row)

    def hessian(self, x, lagrange, obj_factor):
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        return H[self.hs.row, self.hs.col]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):

        msg = "Objective value at iteration #{:d} is - {:g}"
        print(msg.format(iter_count, obj_value))


def main():
    x0 = [1.0, 5.0, 5.0, 1.0]

    lb = [1.0, 1.0, 1.0, 1.0]
    ub = [5.0, 5.0, 5.0, 5.0]

    cl = [25.0, 40.0]
    cu = [2.0e19, 40.0]

    nlp = cyipopt.Problem(n=len(x0), m=len(cl), problem_obj=HS071(), lb=lb,
                          ub=ub, cl=cl, cu=cu)

    nlp.addOption(b'mu_strategy', b'adaptive')
    nlp.addOption(b'tol', 1e-7)

    nlp.setProblemScaling(obj_scaling=2, x_scaling=[1, 1, 1, 1])
    nlp.addOption(b'nlp_scaling_method', b'user-scaling')

    x, info = nlp.solve(x0)

    msg = "Solution of the primal variables: x={:s}\n"
    print(msg.format(repr(x)))

    msg = "Solution of the dual variables: lambda={:s}\n"
    print(msg.format(repr(info['mult_g'])))

    msg = "Objective={:s}\n"
    print(msg.format(repr(info['obj_val'])))


if __name__ == '__main__':
    main()
