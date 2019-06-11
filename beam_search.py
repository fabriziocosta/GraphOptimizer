#!/usr/bin/env python
"""Provides scikit interface."""
import numpy as np
from GraphOptimizer.parallel_utils import parallel_map


class BeamSearch(object):

    def __init__(self,
                 neighborhood=None,
                 objective_func=None,
                 n_iter=100,
                 beam_size=10,
                 parallel=False):
        self.neighborhood = neighborhood
        self.objective_func = objective_func
        self.n_iter = n_iter
        self.beam_size = beam_size
        self.parallel = parallel

    def remove_duplicates(self, graphs):
        _eps_ = 1e-5
        vals = [self.objective_func(g) for g in graphs]
        ids = list(np.argsort(vals).reshape(-1))
        sorted_vals = sorted(vals)
        new_graphs = []
        prev = 1e9
        for val, id in zip(sorted_vals, ids):
            if abs(val - prev) > _eps_:
                new_graphs.append(graphs[id])
            prev = val
        return new_graphs

    def select_beam(self, graphs):
        out_graphs = self.remove_duplicates(graphs)
        sorted_graphs = sorted(
            out_graphs, key=lambda g: self.objective_func(g), reverse=True)
        out_graphs = sorted_graphs[:self.beam_size]
        return out_graphs

    def optimize_single(self, graph):
        out_graphs = [graph]
        for i in range(self.n_iter):
            gen_graphs = []
            for g in out_graphs:
                gen_graphs += self.neighborhood.make_gradient_neighbors(g)
            out_graphs = self.select_beam(gen_graphs)
        return out_graphs

    def _optimize(self, graph, i):
        opt_g = self.optimize_single(graph)
        return (i, opt_g)

    def parallel_optimize(self, graphs):
        out_graphs = []
        for res_graphs in parallel_map(self._optimize, graphs):
            out_graphs += res_graphs
        return out_graphs

    def serial_optimize(self, graphs):
        out_graphs = []
        for graph in graphs:
            out_graphs += self.optimize_single(graph)
        return out_graphs

    def optimize(self, graphs):
        if self.parallel:
            out_graphs = self.parallel_optimize(graphs)
        else:
            out_graphs = self.serial_optimize(graphs)
        out_graphs = self.select_beam(out_graphs)
        return out_graphs


class BeamRandomSearch(object):

    def __init__(self,
                 neighborhood=None,
                 objective_func=None,
                 n_iter=100,
                 beam_size=10,
                 parallel=False):
        self.neighborhood = neighborhood
        self.objective_func = objective_func
        self.n_iter = n_iter
        self.beam_size = beam_size
        self.parallel = parallel

    def remove_duplicates(self, graphs):
        _eps_ = 1e-5
        vals = [self.objective_func(g) for g in graphs]
        ids = list(np.argsort(vals).reshape(-1))
        sorted_vals = sorted(vals)
        new_graphs = []
        prev = 1e9
        for val, id in zip(sorted_vals, ids):
            if abs(val - prev) > _eps_:
                new_graphs.append(graphs[id])
            prev = val
        return new_graphs

    def select_beam(self, graphs):
        out_graphs = self.remove_duplicates(graphs)
        sorted_graphs = sorted(
            out_graphs, key=lambda g: self.objective_func(g), reverse=True)
        out_graphs = sorted_graphs[:self.beam_size]
        return out_graphs

    def optimize_single(self, graph):
        out_graphs = [graph]
        for i in range(self.n_iter):
            gen_graphs = []
            for g in out_graphs:
                gen_graphs += self.neighborhood.make_neighbors(g)
            out_graphs = self.select_beam(gen_graphs)
        return out_graphs

    def _optimize(self, graph, i):
        opt_g = self.optimize_single(graph)
        return (i, opt_g)

    def parallel_optimize(self, graphs):
        out_graphs = []
        for res_graphs in parallel_map(self._optimize, graphs):
            out_graphs += res_graphs
        return out_graphs

    def serial_optimize(self, graphs):
        out_graphs = []
        for graph in graphs:
            out_graphs += self.optimize_single(graph)
        return out_graphs

    def optimize(self, graphs):
        if self.parallel:
            out_graphs = self.parallel_optimize(graphs)
        else:
            out_graphs = self.serial_optimize(graphs)
        out_graphs = self.select_beam(out_graphs)
        return out_graphs
