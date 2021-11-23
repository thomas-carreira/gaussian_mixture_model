import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from data_utils import confidence_ellipse


class GMM:
    def __init__(self, X, k=3):
        rand = random.sample(range(0, len(X) - 1), k)
        self.u = X[rand]
        self.pi = np.array([1/k, 1/k, 1/k])
        self.sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=float)
        self.z = np.zeros((len(X), k))
        self.X = X
        self.k = k
        self.Nk = None
        self.loss_hist = []
        self.update_z()

    def get_multi_gauss(self):
        res = np.zeros((len(self.X), self.k))
        for k in range(self.k):
            res[:, k] = multivariate_normal.pdf(self.X, self.u[k], self.sigma[k])
        return res

    def update_z(self):
        mg = self.get_multi_gauss()
        divider = self.pi.dot(mg.T)
        for k in range(self.k):
            self.z[:, k] = (self.pi[k]*mg[:, k])/divider
        self.Nk = np.sum(self.z, axis=0)

    def update_sigma(self):
        for k in range(self.k):
            temp = []
            for i in range(len(self.X)):
                temp.append(self.z[i][k] * (np.outer((self.X[i] - self.u[k]), (self.X[i] - self.u[k]))))
            self.sigma[k] = np.sum(temp, axis=0)/self.Nk[k]

    def update_u(self):
        for k in range(self.k):
            self.u[k] = (self.z[:, k].dot(self.X))/self.Nk[k]

    def update_pi(self):
        self.pi = self.Nk/len(self.X)

    def step(self):
        self.update_sigma()
        self.update_u()
        self.update_pi()
        self.update_z()

    def optim(self, iter=50):
        for i in range(iter):
            self.step()
            loss = self.loss()
            self.loss_hist.append(loss)
        self.plot_gmm()
        # self.plot_loss()

    def loss(self):
        res = np.zeros((len(self.X), self.k))
        for k in range(self.k):
            res[:, k] = multivariate_normal.pdf(self.X, self.u[k], self.sigma[k])
        res = np.sum(res, axis=1)
        res = np.log(res)
        return -np.sum(res)

    def plot_gmm(self):
        fig, ax = plt.subplots(1, 1)

        id_cluster = np.argmax(self.z, axis=1)
        for k in range(self.k):
            X_by_cluster = self.X[id_cluster[:] == k]
            ax.scatter(X_by_cluster[:, 0], X_by_cluster[:, 1], s=10)
            std = np.sqrt(np.sum((X_by_cluster - self.u[k]) ** 2) / len(X_by_cluster))
            confidence_ellipse(X_by_cluster[:, 0], X_by_cluster[:, 1], ax, u=self.u[k], n_std=std, cov=self.sigma[k], edgecolor='red')
            ax.scatter(self.u[k, 0], self.u[k, 1], c='red', s=20)

        plt.show()

    def plot_loss(self):
        plt.plot(self.loss_hist)
        plt.show
