from scipy import linalg
import numpy as np


def solve_PQMatrix(P, Q):
    lambda2, W0 = linalg.eig(filter_matrix(P * Q, 1e-16))
    W0 = np.matrix(W0)
    LAM = np.matrix(np.diag(np.sqrt(lambda2)))
    V0 = Q * W0 * LAM.I
    return W0, V0, LAM


def filter_matrix(matrix, PQ_param):
    real_part = np.real(matrix).copy()
    imag_part = np.imag(matrix).copy()
    real_part[np.abs(real_part) < PQ_param] = 0
    imag_part[np.abs(imag_part) < PQ_param] = 0
    return real_part + 1j * imag_part


def star_product(SMatrix_A, SMatrix_B):
    if SMatrix_A.shape != SMatrix_B.shape:
        raise Exception

    I = np.matrix(np.identity(SMatrix_A.s11.shape[0]))
    A, B = SMatrix_A, SMatrix_B
    s_ab_11 = A.s11 + A.s12 * (I - B.s11 * A.s22).I * B.s11 * A.s21
    s_ab_12 = A.s12 * (I - B.s11 * A.s22).I * B.s12
    s_ab_21 = B.s21 * (I - A.s22 * B.s11).I * A.s21
    s_ab_22 = B.s22 + B.s21 * (I - A.s22 * B.s11).I * A.s22 * B.s12

    SMatrix_AB = SMatrix()
    SMatrix_AB.loadByPart(s_ab_11, s_ab_12, s_ab_21, s_ab_22)
    return SMatrix_AB


class SMatrix(object):
    def __init__(self):
        self.s11 = None
        self.s12 = None
        self.s21 = None
        self.s22 = None
        self.total = None
        self.shape = None

    def loadByTotal(self, matrix):
        self.total = matrix
        self.shape = matrix.shape
        self.decompose_SMatrix()

    def loadByPart(self, s11, s12, s21, s22):
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22
        self.compose_SMatrix()
        self.shape = self.total.shape

    def compose_SMatrix(self):
        self.total = np.bmat([[self.s11, self.s12], [self.s21, self.s22]])

    def decompose_SMatrix(self):
        m, n = self.shape
        self.s11 = self.total[:round(m / 2), :round(n / 2)]
        self.s12 = self.total[:round(m / 2), round(n / 2):]
        self.s21 = self.total[round(m / 2):, :round(n / 2)]
        self.s22 = self.total[round(m / 2):, round(n / 2):]
