#!/usr/bin/env python
"""Provides scikit interface."""


import numpy as np
import networkx as nx
from sklearn import manifold
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from ego.vectorize import vectorize
from scipy.sparse import csc_matrix
from sklearn.feature_selection import RFECV
from sklearn.linear_model import SGDClassifier
from scipy.sparse import diags
from sklearn.preprocessing import normalize


def get_feature_scaling(
        graphs,
        targets,
        decomposition_funcs=None,
        preprocessors=None,
        nbits=11,
        threshold=0.25):

    x = vectorize(
        graphs,
        decomposition_funcs,
        preprocessors=preprocessors,
        nbits=nbits,
        seed=1)
    estimator = SGDClassifier(penalty='elasticnet', tol=1e-3)
    fs = RFECV(estimator, step=.1, cv=3)
    fs.fit(x, targets)
    fs.estimator_.decision_function(fs.transform(x)).reshape(-1)
    importances = fs.inverse_transform(fs.estimator_.coef_).reshape(-1)
    signs = np.sign(importances)
    importances = np.absolute(importances)
    importances = importances / np.max(importances)
    # non linear thresholding to remove least important features
    th = np.percentile(importances, threshold * 100)
    signs[importances < th] = 0
    importances[importances < th] = 0
    return importances, signs


def basis(data, n_components):
    n_instances, n_features = data.shape
    rows, cols = data.nonzero()
    feature_ids = np.unique(cols)
    data_transpose = data[:, feature_ids].todense().T
    graph = make_graph(data_transpose, d=2, order=2,
                       n_neighbors=3, feature_range=(0.1, 0.9))
    basis = embed(graph, n_components=n_components)
    row = []
    col = []
    vals = []
    for bs, f in zip(basis, feature_ids):
        for i, v in enumerate(bs):
            row.append(f)
            col.append(i)
            vals.append(v)
    row = np.array(row)
    col = np.array(col)
    vals = np.array(vals)
    sparse_basis = csc_matrix(
        (vals, (row, col)), shape=(n_features, n_components))
    sparse_orthogonal_basis = normalize(sparse_basis.T).T
    return sparse_orthogonal_basis


def get_basis(graphs, decomposition_funcs=None, preprocessors=None, nbits=11, n_components=2):
    data = vectorize(graphs, decomposition_funcs,
                     preprocessors=preprocessors, nbits=nbits, seed=1)
    sparse_orthogonal_basis = basis(data, n_components)
    return sparse_orthogonal_basis


def get_supervised_basis(graphs, targets, decomposition_funcs=None, preprocessors=None, nbits=11, n_components=2):
    data = vectorize(graphs, decomposition_funcs,
                     preprocessors=preprocessors, nbits=nbits, seed=1)
    sparse_basis = basis(data, n_components)
    threshold = 0.1
    importances, signs = get_feature_scaling(
        graphs, targets, decomposition_funcs, preprocessors, nbits, threshold)
    # select basis rows according to sign (positive)
    # compute average direction of pos basis
    avg_pos_vec = csc_matrix(normalize(sparse_basis[signs >= 0].mean(axis=0)))
    # remove projection on average pos basis
    proj = sparse_basis.dot(avg_pos_vec.T) * avg_pos_vec
    new_neg_basis = normalize((sparse_basis - proj).T).T
    # select basis elements according to sign
    new_basis = csc_matrix(sparse_basis.shape)
    for i, s in enumerate(signs):
        if s > 0:
            new_basis[i] = sparse_basis[i]
        elif s < 0:
            new_basis[i] = new_neg_basis[i]
    return new_basis
    # scaled_basis = diags(importances).dot(new_basis)
    # return scaled_basis


def transform(graphs, basis=None, decomposition_funcs=None, preprocessors=None, nbits=11):
    data = vectorize(graphs, decomposition_funcs,
                     preprocessors=preprocessors, nbits=nbits, seed=1)
    return np.dot(data, basis).todense()


def estimate_distances(data, d=2):
    if data.ndim == 1:
        data_ = data.reshape(-1, 1)
    else:
        data_ = data
    func = lambda u, v: np.power((np.absolute(u - v)**d).sum(), 1.0 / d)
    distance_mtx = squareform(pdist(data_, func))
    return distance_mtx


def all_distances_graph(data, d=2):
    n = data.shape[0]
    h = nx.Graph()
    h.add_nodes_from(range(n))
    distance_mtx = estimate_distances(data, d=d)
    for i, row in enumerate(distance_mtx):
        for j, val in enumerate(row):
            if i != j:
                h.add_edge(i, j, weight=val, len=val)
    return h


def knn_graph(g, n_neighbors=3):
    h = nx.Graph()
    h.add_nodes_from(g.nodes())
    for u in g.nodes():
        ns = g.neighbors(u)
        for v in sorted(ns, key=lambda m: g.edges[u, m]['weight'])[:n_neighbors]:
            w = g.edges[u, v]['weight']
            h.add_edge(u, v, weight=w, len=w)
    return h


def iterated_minimum_spanning_tree(g, order=2):
    h = nx.Graph(g)
    t = nx.minimum_spanning_tree(h)
    for i in range(order - 1):
        h.remove_edges_from(t.edges())
        t2 = nx.minimum_spanning_tree(h)
        t = nx.compose(t, t2)
    return t


def make_graph(data, d=2, order=2, n_neighbors=3, feature_range=(0.1, 0.9)):
    g = all_distances_graph(data, d=d)
    t = iterated_minimum_spanning_tree(g, order=order)
    k = knn_graph(g, n_neighbors=n_neighbors)
    graph = nx.compose(t, k)
    # rescale edge lenghts
    vals = minmax_scale([graph.edges[u, v]['len']
                         for u, v in graph.edges()], feature_range=feature_range)
    for val, (u, v) in zip(vals, graph.edges()):
        graph.edges[u, v]['len'] = val
        graph.edges[u, v]['weight'] = val
    return graph


def embed_graph_mds(graph, n_components=2, weight='len'):
    p = dict(nx.shortest_path_length(graph, weight=weight))
    mapper = {u: i for i, u in enumerate(graph.nodes())}
    dissimilarity = np.zeros((len(graph), len(graph)))
    for i in graph.nodes():
        ii = mapper[i]
        for j in graph.nodes():
            jj = mapper[j]
            if p[i].get(j, None) is not None:
                dissimilarity[ii, jj] = p[i][j]
    mds = manifold.MDS(
        n_components=n_components,
        dissimilarity="precomputed")
    x = mds.fit(dissimilarity).embedding_
    return x


def canonical_direction(x):
    u, s, vh = np.linalg.svd(x, full_matrices=True)
    z = np.dot(x, vh.T)
    return z


def scale_largest_dim(x):
    xmin = np.min(x, axis=0).reshape(-1)
    xmax = np.max(x, axis=0).reshape(-1)
    xlen = np.absolute(xmax - xmin).max()
    x = (x - xmin) / xlen
    return x


def embed(graph, n_components=2, weight='len'):
    if n_components > 2:
        x = embed_graph_mds(graph, n_components, weight)
        x = canonical_direction(x)
    elif n_components == 2:
        pos = nx.kamada_kawai_layout(graph, weight=weight)
        x = np.array([list(pos[i]) for i in pos])
        x = canonical_direction(x)
        x = x.reshape(-1, 2)
    elif n_components == 1:
        x = embed_graph_mds(graph, n_components, weight)
        x = x.reshape(-1, 1)
    x = scale_largest_dim(x)
    return x
