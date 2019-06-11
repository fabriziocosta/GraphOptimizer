#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx

from collections import defaultdict

from sklearn.feature_selection import RFECV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestRegressor
import GraphOptimizer.treeinterpreter as ti

from ego.vectorize import set_feature_size
from ego.encode import make_encoder
from ego.vectorize import vectorize_graphs


class PartImportanceEstimator(object):

    def __init__(self,
                 decompose_func=None,
                 preprocessor=None,
                 nbits=14):
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        encoding_func = make_encoder(
            decompose_func, preprocessors=preprocessor, bitmask=self.bitmask, seed=1)
        self.encoding_func = encoding_func
        self.estimator = None

    def feature_importance(self, graphs):
        x = vectorize_graphs(graphs, encoding_func=self.encoding_func,
                             feature_size=self.feature_size)
        prediction, biases, contributions = ti.predict(self.estimator, x)
        importances = np.mean(contributions, axis=0)
        intercept = biases[0]
        importance_dict = dict(enumerate(importances))
        return importance_dict, intercept

    def fit(self, graphs, targets):
        x = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        self.estimator = RandomForestRegressor(n_estimators=100)
        self.estimator = self.estimator.fit(x, targets)
        self.importance_dict, self.intercept = self.feature_importance(graphs)
        return self

    def node_and_edge_importance(self, graph):
        importance_dict, intercept = self.feature_importance([graph])
        # remove intercept: we are interested in the relative
        # importance of features, i.e. pos and neg contributions
        # adding the intercept would just offset values and mask
        # pos neg contribution
        intercept = 0
        node_scores = defaultdict(float)
        edge_scores = defaultdict(float)
        codes, fragments = self.encoding_func(graph)
        n_fragments = float(len(fragments))
        for code, fragment in zip(codes, fragments):
            n_nodes_in_fragment = float(nx.number_of_nodes(fragment))
            n_edges_in_fragment = float(nx.number_of_edges(fragment))
            for u in fragment.nodes():
                node_scores[u] += (importance_dict[code] +
                                   intercept / n_fragments) / n_nodes_in_fragment
            for u, v in fragment.edges():
                if u > v:
                    u, v = v, u
                edge_scores[(u, v)] += (importance_dict[code] +
                                        intercept / n_fragments) / n_edges_in_fragment
        return node_scores, edge_scores

    def node_importance(self, graph):
        node_scores, edge_scores = self.node_and_edge_importance(graph)
        return node_scores

    def predict(self, graph):
        # return node and edge importance
        node_score_dict, edge_score_dict = self.node_and_edge_importance(graph)
        return node_score_dict, edge_score_dict

    def score(self, graph):
        node_score_dict, edge_score_dict = self.node_and_edge_importance(graph)
        node_score = np.sum([val for val in node_score_dict.values()])
        return node_score

    def decision_function(self, graphs):
        x = vectorize_graphs(
            graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        return self.estimator.predict(x)


class PartImportanceClassificationEstimator(object):

    def __init__(self,
                 to_binary_classification=None,
                 decompose_func=None,
                 preprocessor=None,
                 nbits=14):
        self.to_binary_classification = to_binary_classification
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        encoding_func = make_encoder(
            decompose_func, preprocessors=preprocessor, bitmask=self.bitmask, seed=1)
        self.encoding_func = encoding_func
        self.estimator = None

    def feature_importance(self, pos_graphs, neg_graphs):
        graphs = pos_graphs + neg_graphs
        y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        x = vectorize_graphs(graphs,
                             encoding_func=self.encoding_func,
                             feature_size=self.feature_size)
        estimator = SGDClassifier(penalty='elasticnet', tol=1e-3)
        fs = RFECV(estimator, step=.1, cv=3)
        fs.fit(x, y)
        fs.estimator_.decision_function(fs.transform(x)).reshape(-1)
        self.estimator = fs.estimator_
        importances = fs.inverse_transform(fs.estimator_.coef_).reshape(-1)
        intercept = fs.estimator_.intercept_[0]
        importance_dict = dict(enumerate(importances))
        return importance_dict, intercept

    def classifier_fit(self, pos_graphs, neg_graphs):
        self.importance_dict, self.intercept = self.feature_importance(
            pos_graphs, neg_graphs)
        return self

    def classifier_decision_function(self, graphs):
        x = vectorize_graphs(graphs,
                             encoding_func=self.encoding_func,
                             feature_size=self.feature_size)
        preds = self.estimator.decision_function(x)
        return preds

    def classifier_predict(self, graphs):
        x = vectorize_graphs(graphs,
                             encoding_func=self.encoding_func,
                             feature_size=self.feature_size)
        preds = self.estimator.predict(x)
        return preds

    def fit(self, graphs, targets):
        # transform problem into binary classification of Pareto shells
        pos_graphs, neg_graphs = self.to_binary_classification(graphs, targets)
        # learn predictive model and extract feature importance
        return self.classifier_fit(pos_graphs, neg_graphs)

    def node_and_edge_importance(self, graph):
        node_scores = defaultdict(float)
        edge_scores = defaultdict(float)
        codes, fragments = self.encoding_func(graph)
        n_fragments = float(len(fragments))
        for code, fragment in zip(codes, fragments):
            n_nodes_in_fragment = float(nx.number_of_nodes(fragment))
            n_edges_in_fragment = float(nx.number_of_edges(fragment))
            for u in fragment.nodes():
                node_scores[u] += (self.importance_dict[code] +
                                   self.intercept / n_fragments) / n_nodes_in_fragment
            for u, v in fragment.edges():
                if u > v:
                    u, v = v, u
                edge_scores[(u, v)] += (self.importance_dict[code] +
                                        self.intercept / n_fragments) / n_edges_in_fragment
        return node_scores, edge_scores

    def predict(self, graph):
        # return node and edge importance
        node_score_dict, edge_score_dict = self.node_and_edge_importance(graph)
        return node_score_dict, edge_score_dict

    def score(self, graph):
        node_score_dict, edge_score_dict = self.node_and_edge_importance(graph)
        node_score = np.sum([val for val in node_score_dict.values()])
        return node_score
