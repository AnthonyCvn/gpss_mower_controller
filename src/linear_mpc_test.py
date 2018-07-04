#!/usr/bin/env python
import numpy as np
import quadprog
import cvxopt


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M)  # quick way to build a symmetric matrix
q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([3., 2., -2.]).reshape((3,))

print "QP solution:", quadprog_solve_qp(P, q, G, h)
