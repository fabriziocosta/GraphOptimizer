#!/usr/bin/env python
"""Provides scikit interface."""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import numpy as np
import logging

logger = logging.getLogger()


class ExpectedImprovementEstimator(object):
    """ExpectedImprovementEstimator."""

    def __init__(self):
        """init."""
        kernel = 1.0 * Matern(length_scale=0.001, nu=1.5) + \
            1.0 * RationalQuadratic(length_scale=0.001, alpha=0.1)
        # + WhiteKernel(noise_level=0.01)
        self.gp_estimator = GaussianProcessRegressor(
            n_restarts_optimizer=10, kernel=kernel)
        self.loss_optimum = 0

    def fit(self, x, y):
        """fit."""
        _num_attempts_ = 10
        self.loss_optimum = max(y)
        logger.debug('Current optimum: %.4f' % self.loss_optimum)
        for it in range(_num_attempts_):
            try:
                self.gp_estimator.fit(x, y)
            except Exception as e:
                logger.debug('Error: %s' % e)
            else:
                break
        loglike = self.gp_estimator.log_marginal_likelihood(
            self.gp_estimator.kernel_.theta)
        logger.debug("Posterior (kernel: %s)\n Log-Likelihood: %.3f (%.3f)" % (
            self.gp_estimator.kernel_, loglike, loglike / len(y)))

    def predict_mean_and_variance(self, x):
        """predict_mean_and_variance."""
        mu, sigma = self.gp_estimator.predict(x, return_std=True)
        return mu, sigma

    def predict(self, x):
        """predict."""
        mu, sigma = self.predict_mean_and_variance(x)
        with np.errstate(divide='ignore'):
            z = (mu - self.loss_optimum) / sigma
            a = (mu - self.loss_optimum) * norm.cdf(z)
            b = sigma * norm.pdf(z)
            exp_imp = a + b
            exp_imp[sigma == 0.0] = 0
        return exp_imp


class GraphExpectedImprovementEstimator(object):
    """GraphExpectedImprovementEstimator."""

    def __init__(self, graph_vector_embedder=None):
        """init."""
        self.exp_imp_estimator = ExpectedImprovementEstimator()
        self.graph_vector_embedder = graph_vector_embedder

    def fit(self, graphs, targets):
        """fit."""
        self.graph_vector_embedder.fit(graphs, targets)
        # self.graph_vector_embedder.fit(graphs)
        x = self.graph_vector_embedder.transform(graphs)
        self.exp_imp_estimator.fit(x, targets)
        return self

    def transform(self, graphs):
        """transform."""
        x = self.graph_vector_embedder.transform(graphs)
        return x

    def expected_improvement(self, x):
        """expected_improvement."""
        return self.exp_imp_estimator.predict(x)

    def predict(self, graphs):
        """predict."""
        x = self.graph_vector_embedder.transform(graphs)
        y = self.exp_imp_estimator.predict(x)
        return y
