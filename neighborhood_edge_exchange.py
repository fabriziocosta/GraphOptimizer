#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
from random import choice
from toolz.curried import pipe, map


class NeighborhoodEdgeExchange(object):

    def __init__(self,
                 perturbation_size=2,
                 part_importance_estimator=None,
                 objective_func=None,
                 n_alternatives=3):
        self.perturbation_size = perturbation_size
        self.part_importance_estimator = part_importance_estimator
        self.objective_func = objective_func
        self.n_alternatives = n_alternatives

    def fit(self, graphs):
        pass
        return self

    def __repr__(self):
        return 'EdgeExchangeNeighborhood'

    def max_connected_component_size(self, graph):
        return pipe(graph, nx.connected_components, map(lambda x: len(x)), max)

    def select_edge_at_random(self, graph):
        u, v = choice(list(graph.edges()))
        return u, v

    def select_edge_with_bias(self, graph):
        _eps_ = 1e-4
        # select root nodes with probability proportional to negative scores
        res = self.part_importance_estimator.predict(graph)
        node_score_dict, edge_score_dict = res
        s = np.array([node_score_dict[u] + _eps_ for u in graph.nodes()])
        probs = np.exp(-s / np.max(np.absolute(s)))
        probs = probs / probs.sum()
        ns = [u for u in graph.nodes()]
        sel_u = np.random.choice(ns, size=1, p=probs)[0]

        # extract score of neighbor edges
        es = np.array([edge_score_dict[sel_u, v] + _eps_
                       for v in graph.neighbors(sel_u)])
        eprobs = np.exp(-es / np.max(np.absolute(es)))
        eprobs = eprobs / eprobs.sum()
        vs = [u for u in graph.neighbors(sel_u)]
        sel_v = np.random.choice(vs, size=1, p=eprobs)[0]
        return sel_u, sel_v

    def edge_exchange(self, graph, u1, v1, u2, v2):
        g = nx.Graph(graph)
        d1 = g.edges[u1, v1]
        d2 = g.edges[u2, v2]
        # remove edges
        g.remove_edge(u1, v1)
        g.remove_edge(u2, v2)
        # rewire edges exchanging endpoints
        g.add_edge(u1, v2)
        g.edges[u1, v2].update(d1)
        g.add_edge(u2, v1)
        g.edges[u2, v1].update(d2)
        return g

    def perturb(self, orig_graph):
        g = nx.Graph(orig_graph)
        for i in range(self.perturbation_size):
            u1, v1 = self.select_edge_at_random(g)
            u2, v2 = self.select_edge_at_random(g)
            # redo edge selection if e2 is the same as e1
            while (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                u2, v2 = self.select_edge_at_random(g)
            g = self.edge_exchange(g, u1, v1, u2, v2)
        return g

    def biased_perturb(self, orig_graph):
        g = nx.Graph(orig_graph)
        for i in range(self.perturbation_size):
            u1, v1 = self.select_edge_with_bias(g)
            u2, v2 = self.select_edge_at_random(g)
            # redo edge selection if e2 is the same as e1
            while (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                u2, v2 = self.select_edge_at_random(g)
            g = self.edge_exchange(g, u1, v1, u2, v2)
        return g

    def gradient_descent(self, graph):
        out_graphs = [graph]
        for i in range(self.n_alternatives):
            out_graphs.append(self.biased_perturb(graph))
        # sort graphs according to average objective score
        # (to contrast the tendency to produce large graphs)
        func = lambda g: self.objective_func(g) / float(len(g))
        sorted_graphs = sorted(out_graphs, key=func, reverse=True)
        selected_sorted_graphs = sorted_graphs[:5]

        # select graph that maximizes the average importance score
        # (to contrast the tendency to produce large graphs)
        func = lambda g: self.part_importance_estimator.score(
            g) / float(len(g))
        best_graph = max(selected_sorted_graphs, key=func)
        return best_graph
