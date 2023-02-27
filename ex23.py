# CFS Code for GEPP (including the rank evaluation for judging if there is a unique solution)
import numpy as np
from numpy import matrix as MX
from numpy import linalg as LA
from numpy import array as AR


def pp(A, k):
    '''
    This function takes in a matrix A, and column k and
    perform partial pivoting (PP) for k-th column and
    returns the resulting matrix, A
    '''
    M = np.argmax(abs(A[k:, k])) + k
    A[[k, M], :] = A[[M, k], :]
    return A


def ref(A, s):
    '''
    This function takes in a matrix A and and option for GE, s (if s == 1: GEPP, else: naive GE)
    returns an augmented matrix in ref by using Gaussian elimination algorithm (GE)
    '''
    n, col = np.shape(A)
    for k in range(n - 1):
        if s == 1:
            A = pp(A, k)
        for i in range(k + 1, n):
            if A[i, k] != 0:
                lam = A[i, k] / A[k, k]
                A[i, k:] = A[i, k:] - lam * A[k, k:]
    return A


def info(A, b):
    '''
    This function takes in a coeff matrix A and const vector b
    returns a matrix info of the augmented matrix (Ab = [A, b]), the number of rows (n),
    and the number of solutions (n_sol) to [A, b] by rank analysis
    '''
    n, no_use = np.shape(A)
    Ab = np.hstack([A, b])
    rank_Ab = LA.matrix_rank(Ab);
    rank_A = LA.matrix_rank(A)
    if rank_Ab - rank_A == 1:
        n_sol = 0
    elif rank_Ab - rank_A != 0:
        raise Exception('Error! Something is wrong.')
    else:
        if rank_A < n:
            n_sol = np.inf
        elif rank_A > n:
            raise Exception('Error! Something is wrong.')
        else:
            n_sol = 1
    return Ab, n, n_sol


def GE(A, b, s):
    '''
    This function takes in a coeff matrix A, const vector b, and option for GE, s (if s == 1: GEPP, else: naive GE)
    returns a unique solution (if it exists) by using back substitution and REF
    '''
    Ab, n, n_sol = info(A, b)
    if n_sol == 1:
        Uc = ref(Ab, s)
        if s == 1:
            print('Solution (by GEPP algorithm) is: ')
        else:
            print('Solution (by naive GE algorithm) is: ')
        x = np.zeros((n, 1))
        for k in range(n - 1, -1, -1):
            x[k, 0] = (Uc[k, n] - np.matmul(Uc[k, k + 1:n], x[k + 1:n, 0])) / Uc[k, k]
        return x
    elif n_sol == 0:
        print('This linear system is inconsistent. (No solution exists.) ')
    else:
        print('This linear system is consistent but dependent. (Infinitely many solutions exist.) ')


def LUDD(A, b):
    '''
    takes in A, b and return L, U, b_prime by Doolittle algorithm
    L, U, b_prime must be matrix form
    '''

    pass

    return L, U, b_prime


def FSBS(A, b):
    '''
    takes in A, b and call LUDD(A, b)
    return [L,pb], y, x
    '''

    Ab, n, n_sol = ch2.info(A, b)

    if n_sol == 1:
        L, U, b_prime = LUDD(A, b)
        Lbp = np.hstack([L, b_prime])
        y = np.zeros((n, 1))

        for k in range(0, n, 1):
            y[k, 0] = (Lbp[k, n] - np.matmul(Lbp[k, 0:k], y[0:k, 0])) / Lbp[k, k]

        Uy = np.hstack([U, y])
        x = np.zeros((n, 1))
        for k in range(n - 1, -1, -1):
            x[k, 0] = (Uy[k, n] - np.matmul(Uy[k, k + 1:n], x[k + 1:n, 0])) / Uy[k, k]

        return Lbp, y, x
    else:
        print('no_solution')


A = AR([[4., 3, -5], [-2, -4, 5], [8, 8, 0]])
b = AR([[-7], [11], [-8]])

LUDD(A, b)
##FSBS(A,b)
##LA.solve(A,b)