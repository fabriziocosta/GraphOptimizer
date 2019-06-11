#!/usr/bin/env python
"""Provides scikit interface."""
import networkx as nx
from GraphOptimizer.parallel_utils import parallel_map


class IteratedLocalSearch(object):

    def __init__(self,
                 neighborhood=None,
                 objective_func=None,
                 n_iter_per_local_search=10,
                 n_local_min_hops=3,
                 parallel=False):
        self.neighborhood = neighborhood
        self.objective_func = objective_func
        self.n_iter_per_local_search = n_iter_per_local_search
        self.n_local_min_hops = n_local_min_hops
        self.parallel = parallel

    def local_search(self, orig_graph):
        best_g = nx.Graph(orig_graph)
        for i in range(self.n_iter_per_local_search):
            candidate_g = self.neighborhood.gradient_descent(best_g)
            if self.accept(candidate_g, best_g):
                best_g = nx.Graph(candidate_g)
        return best_g

    def accept(self, candidate_graph, best_graph):
        candidate_score = self.objective_func(candidate_graph)
        best_score = self.objective_func(best_graph)
        if candidate_score > best_score:
            return True
        return False

    def optimize_single(self, graph):
        best_graph = graph
        for i in range(self.n_local_min_hops):
            if i == 0:
                candidate_graph = best_graph
            else:
                candidate_graph = self.neighborhood.perturb(best_graph)
            candidate_graph = self.local_search(candidate_graph)
            if self.accept(candidate_graph, best_graph):
                best_graph = nx.Graph(candidate_graph)
        return best_graph

    def _optimize(self, graph, i):
        opt_g = self.optimize_single(graph)
        return (i, opt_g)

    def parallel_optimize(self, graphs):
        out_graphs = parallel_map(self._optimize, graphs)
        return out_graphs

    def serial_optimize(self, graphs):
        out = []
        for i, graph in enumerate(graphs):
            opt_graph = self.optimize_single(graph)
            out.append(opt_graph)
        return out

    def optimize(self, graphs):
        if self.parallel:
            return self.parallel_optimize(graphs)
        else:
            return self.serial_optimize(graphs)
