# -*- coding: utf-8 -*-
"""
    Input:
        Hermitian matrix: A

    Output:
        Eigendecomposition of matrix A: A = Ch * Z * C
"""
import numpy as np
import copy as cp
import time


def time_count(f):
    def wrapped_f(*args):
        start_time = time.time()
        out = f(*args)
        end_time = time.time()
        duration = end_time - start_time
        print('Time of executing of %s = %.6f sec.' % (f.__name__, duration))
        return out
    return wrapped_f


def check_hermite(A):
    AT = A.transpose()
    A_ = A.conj()
    if (AT == A_).all():
        return True
    else:
        return False


def rotation_left(M, n, theta, p, q):
    RM = cp.deepcopy(M)
    for k in range(n):
        RM[p, k] = (1 / np.sqrt(2)) * (M[p, k] * np.exp(-theta * 1j) - M[q, k] * np.exp(theta * 1j))
        RM[q, k] = (1 / np.sqrt(2)) * (M[p, k] * np.exp(-theta * 1j) + M[q, k] * np.exp(theta * 1j))
    return RM


def rotation_right(M, n, theta, p, q):
    MR = cp.deepcopy(M)
    for k in range(n):
        MR[k, p] = (1 / np.sqrt(2)) * (M[k, p] * np.exp(theta * 1j) - M[k, q] * np.exp(-theta * 1j))
        MR[k, q] = (1 / np.sqrt(2)) * (M[k, p] * np.exp(theta * 1j) + M[k, q] * np.exp(-theta * 1j))
    return MR


@time_count
def diagonalize_jacobi_hermite(A, n, epsilon=0.000001):
    # the main part: diagonalization of hermitian matrix
    A_ = cp.deepcopy(A)
    U = np.eye(n, dtype=complex)
    non_diag = dict()
    for i in range(n):
        for j in range(n):
            if i != j:
                non_diag[(i, j)] = np.abs(A_[i, j])
    value = sum(non_diag.values())
    while value > epsilon:
        ind = max(non_diag, key=non_diag.get)
        p = min(ind)
        q = max(ind)
        try:
            phi_1 = np.arctan(np.imag(A_[p, q]) / np.real(A_[p, q]))
        except ZeroDivisionError:
            print('Divided by zero: np.real(A_[p, q]')
            phi_1 = 0
        try:
            phi_2 = np.arctan(2 * np.abs(A_[p, q]) / (A_[p, p] - A_[q, q]))
        except ZeroDivisionError:
            print('Divided by zero: (A_[p, p] - A_[q, q])')
            phi_2 = 0
        theta_1 = (2 * phi_1 - np.pi) / 4
        theta_2 = phi_2 / 2
        RM = rotation_left(A_, n, theta_1, p, q)
        RRM = rotation_left(RM, n, theta_2, p, q)
        MR = rotation_right(RRM, n, theta_1, p, q)
        A_ = rotation_right(MR, n, theta_2, p, q)
        RR = np.eye(n, dtype=complex)
        RR[p, p] = - 1j * np.exp(-theta_1 * 1j) * np.sin(theta_2)
        RR[p, q] = - np.exp(theta_1 * 1j) * np.cos(theta_2)
        RR[q, p] = np.exp(-theta_1 * 1j) * np.cos(theta_2)
        RR[q, q] = 1j * np.exp(theta_1 * 1j) * np.sin(theta_2)
        U = RR.dot(U)
        for i in range(n):
            for j in range(n):
                if i != j:
                    non_diag[(i, j)] = np.abs(A_[i, j])
        value = sum(non_diag.values())

    return U, A_


@time_count
def get_eigen2x2(A):
    a11 = A[0, 0]
    a22 = A[1, 1]
    trace = a11 + a22
    det = a11 * a22 - A[1, 0] * A[0, 1]
    lambda1 = (trace + np.sqrt(trace ** 2 - 4 * det)) / 2
    lambda2 = (trace - np.sqrt(trace ** 2 - 4 * det)) / 2
    X1 = np.array([[a11 - lambda2], [A[1, 0]]])
    X2 = np.array([[A[0, 1]], [a22 - lambda1]])
    try:
        X1 /= np.linalg.norm(X1)
    except ZeroDivisionError:
        X1 = np.zeros((2, 1))
    try:
        X2 /= np.linalg.norm(X2)
    except ZeroDivisionError:
        X2 = np.zeros((2, 1))
    C = np.concatenate((X1, X2), axis=1)
    T = np.array([[lambda1, 0], [0, lambda2]])
    return C, T


def decompose(A):
    (n, n) = np.shape(A)
    if n > 2:
        C, Z = diagonalize_jacobi_hermite(A, n)
    else:
        C, Z = get_eigen2x2(A)

    A__ = C.transpose().conj().dot(Z).dot(C)
    print('Matrix A = Ch * Z * C \n', np.around(A__, decimals=2))
    print('Matrix C:\n', np.around(C, decimals=2))
    print('Matrix Z:\n', np.around(Z, decimals=2))
    eigenvalue = []
    for i in range(n):
        eigenvalue.append(Z[i, i])
    print('Eigenvalues given by Jacobi method: \n', eigenvalue)
    print('Numpy eigenvalues: \n', np.linalg.eigvals(A))
    return


def main():
    A = np.array([[1, 1 + 5j, 2 + 4j, 5, 10],
                  [1 - 5j, 2, 6, 1 - 2j, 7],
                  [2 - 4j, 6, 3, 3 - 1j, 1],
                  [5, 1 + 2j, 3 + 1j, 2, 5 - 2j],
                  [10, 7, 1, 5 + 2j, 1]])

    # A = np.array([[1, 2, 3, 4],
    #               [2, 7, 2, 5],
    #               [3, 2, 3, 3],
    #               [4, 5, 3, 2]], dtype=complex)

    # A = np.array([[1, 2 + 4j],
    #              [2 - 4j, 5]])

    if check_hermite(A):
        decompose(A)
    else:
        print("This matrix is not hermitian")
        return


if __name__ == '__main__':
    main()
