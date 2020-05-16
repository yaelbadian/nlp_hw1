import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import fmin_l_bfgs_b


class Optimization:
    def __init__(self, mat, list_of_mats, lamda):
        self.mat = mat
        self.list_of_mats = list_of_mats
        self.lamda = lamda

    def calculate_linear_term(self, v):
        return self.mat.dot(v).sum()

    def calculate_exps(self, v):
        exps = []
        for mat in self.list_of_mats:
            exps.append(np.exp(mat.dot(v)))
        return exps

    def calculate_sum_exps(self, exps):
        sum_of_exps = np.zeros(self.mat.shape[0]).T
        for exp in exps:
            sum_of_exps += exp
        return sum_of_exps

    @staticmethod
    def calculate_normalization_term(sum_of_exps):
        if sum_of_exps[sum_of_exps == 0].shape[0] > 1:
            print(sum_of_exps)
            print(np.log(sum_of_exps).sum())
        return np.log(sum_of_exps).sum()

    def calculate_regularization_term(self, v):
        return 0.5 * self.lamda * (v**2).sum()

    def calculate_empirical_counts(self):
        return self.mat.sum(axis=0)

    def calculate_expected_counts(self, exps, sum_of_exps):
        sum_of_mats = csr_matrix(np.zeros(self.mat.shape)).T
        for mat, exp in zip(self.list_of_mats, exps):
            sum_of_mats += csr_matrix.multiply(mat.T, exp)
        return (csr_matrix.multiply(sum_of_mats, (1 / sum_of_exps))).sum(axis=1).T

    def calculate_regularization_grad(self, v):
        return self.lamda * v

    def calc_objective_per_iter(self, v):
        exps = self.calculate_exps(v)
        sum_of_exps = self.calculate_sum_exps(exps)
        linear_term = self.calculate_linear_term(v)
        normalization_term = self.calculate_normalization_term(sum_of_exps)
        regularization_term = self.calculate_regularization_term(v)
        empirical_counts = self.calculate_empirical_counts()
        expected_counts = self.calculate_expected_counts(exps, sum_of_exps)
        regularization_grad = self.calculate_regularization_grad(v)
        likelihood = linear_term - normalization_term - regularization_term
        grad = empirical_counts - expected_counts - regularization_grad
        # print('likelihood:', likelihood, 'norm grad', np.linalg.norm(grad))
        return (-1) * likelihood, (-1) * grad

    @staticmethod
    def init_weights(k):
        return np.random.normal(size=k)

    @staticmethod
    def optimize_weights(mat, list_of_mats, v=None, lamda=10):
        opt = Optimization(mat, list_of_mats, lamda)
        if v is None:
            v = Optimization.init_weights(mat.shape[1])
        optimal_params = fmin_l_bfgs_b(func=opt.calc_objective_per_iter, x0=v, maxiter=200, iprint=50)
        return optimal_params[0], optimal_params[1]