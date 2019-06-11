#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
from eden.graph import vectorize as eden_vectorize
from ego.vectorize import vectorize as ego_vectorize
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from umap import UMAP
from GraphOptimizer.neighborhood_graph_grammar import NeighborhoodGraphGrammar
from GraphOptimizer.neighborhood_graph_grammar import NeighborhoodEgoGraphGrammar
from toolz import curry

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

import logging
logger = logging.getLogger()


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


def remove_duplicates(graphs):
    x = eden_vectorize(graphs, complexity=2, nbits=10).todense()
    vals = x.sum(axis=1).T.A[0]
    sorted_ids = np.argsort(vals)
    sorted_vals = np.sort(vals)
    prev = sorted_vals[0]
    sel_ids = [0]
    for sorted_id, sorted_val in zip(sorted_ids, sorted_vals):
        if prev != sorted_val:
            sel_ids.append(sorted_id)
        prev = sorted_val
    sel_graphs = [graphs[sel_id] for sel_id in sel_ids]
    return sel_graphs


class AutoEncoder(object):

    def __init__(
            self,
            input_dim=1024,
            dim_enc_layer_1=256,
            dim_enc_layer_2=128,
            encoding_dim=64,
            dim_dec_layer_1=128,
            dim_dec_layer_2=256,
            batch_size=256,
            epochs=10000,
            patience=50,
            min_delta=0,
            log_dir=None,
            verbose=0):
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta

        self.log_dir = log_dir
        self.verbose = verbose

        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(dim_enc_layer_1, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(dim_enc_layer_2, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(self.encoding_dim, activation='sigmoid')(encoded)

        decoded = Dense(dim_dec_layer_1, activation='relu')(encoded)
        decoded = Dense(dim_dec_layer_2, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adadelta',
                                 loss='binary_crossentropy')

        # this model maps an input to its encoded representation
        self.encoder = Model(input_layer, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        deco = self.autoencoder.layers[-3](encoded_input)
        deco = self.autoencoder.layers[-2](deco)
        deco = self.autoencoder.layers[-1](deco)
        # create the decoder model
        self.decoder = Model(encoded_input, deco)

    def fit(self, x_train, x_val=None):
        callbacks = []
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=self.patience, min_delta=self.min_delta))
        if self.log_dir is not None:
            callbacks.append(TensorBoard(log_dir=self.log_dir))
        if x_val is None:
            x_val = x_train
        self.autoencoder.fit(
            x_train, x_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(x_val, x_val),
            callbacks=callbacks,
            verbose=self.verbose)
        return self

    def transform(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        encods = self.encoder.predict(x)
        decods = self.decoder.predict(encods)
        return decods


class Embedder(object):

    def __init__(
            self,
            decomposition_funcs=None,
            preprocessors=None,
            nbits=12,
            seed=1,
            target_metric='categorical',
            n_components_mid=1024,
            n_components_low=32,
            n_components=2,
            embed_n_neighbors=10,
            core=2,
            context=2,
            count=2,
            grammar_n_neighbors=None,
            n_iter=0):
        self.vectorize = curry(ego_vectorize)(
            decomposition_funcs=decomposition_funcs,
            preprocessors=preprocessors,
            nbits=nbits,
            seed=seed)
        self.n_components_mid = n_components_mid
        self.svd = TruncatedSVD(n_components=n_components_mid)
        self.minmaxscaler = MinMaxScaler()
        self.normalizer = Normalizer()
        self.n_components_low = n_components_low
        self.autoencoder = AutoEncoder(input_dim=n_components_mid,
                                       encoding_dim=n_components_low)

        self.target_metric = target_metric
        self.umap = UMAP(
            n_components=n_components,
            n_neighbors=embed_n_neighbors,
            target_metric=target_metric)
        self.neigh = NeighborhoodGraphGrammar(
            core=core, context=context, count=count, n_neighbors=grammar_n_neighbors)
        self.n_iter = n_iter

    def _make_variants(self, graphs):
        neigh_graphs = [new_g for g in graphs
                        for new_g in self.neigh.make_all_neighbors(
                            g, include_original=False)]
        neigh_graphs = remove_duplicates(neigh_graphs)
        out_graphs = neigh_graphs[:]
        for i in range(self.n_iter - 1):
            neigh_graphs = [new_g for g in neigh_graphs
                            for new_g in self.neigh.make_all_neighbors(
                                g, include_original=False)]
            neigh_graphs = remove_duplicates(neigh_graphs)
            out_graphs += neigh_graphs[:]
        return out_graphs

    def fit(self, orig_graphs, original_targets):
        """fit."""
        assert(len(orig_graphs) >= self.n_components_mid), 'Num instances (%d) cannot be smaller than n_components_mid (%d)' % (
            len(orig_graphs), self.n_components_mid)
        if self.target_metric == 'categorical':
            targets = LabelEncoder().fit_transform(original_targets)
        else:
            targets = original_targets
        if self.n_iter == 0:
            y = list(targets)
            graphs = orig_graphs
        else:
            self.neigh.fit(orig_graphs)
            logger.debug('Grammar: %s' % self.neigh)
            semi_graphs = self._make_variants(orig_graphs)
            logger.debug('# generated graphs: %d' % len(semi_graphs))
            y = list(targets) + [-1] * len(semi_graphs)
            graphs = orig_graphs + semi_graphs
        y = np.array(y)
        # graph to high dim sparse
        x = self.vectorize(graphs)

        # high dim sparse to mid dim dense
        self.svd.fit(x)
        x_mid = self.svd.transform(x)
        x_mid = self.minmaxscaler.fit_transform(x_mid)
        x_mid = self.normalizer.fit_transform(x_mid)

        # mid dim to lower dim using non linear mapping
        x_nonlin = self.autoencoder.fit(x_mid).transform(x_mid)

        # lower dim dense to low dim
        y = np.array(y)
        self.umap.fit(x_nonlin, y)
        x_low = self.umap.transform(x_nonlin)

        # rotation
        u, s, vh = np.linalg.svd(x_low, full_matrices=True)
        self.rotation = vh.T
        x_low_rot = np.dot(x_low, self.rotation)

        # scale
        self.xmin = np.min(x_low_rot, axis=0).reshape(-1)
        self.xmax = np.max(x_low_rot, axis=0).reshape(-1)
        self.xlen = np.absolute(self.xmax - self.xmin).max()

        return self

    def _rotate_and_scale(self, x):
        x_low_rot = np.dot(x, self.rotation)
        x_low_rot_rescaled = (x_low_rot - self.xmin) / self.xlen
        return x_low_rot_rescaled

    def transform(self, graphs):
        """transform."""
        x = self.vectorize(graphs)
        x_mid = self.svd.transform(x)
        x_mid = self.minmaxscaler.transform(x_mid)
        x_mid = self.normalizer.transform(x_mid)
        x_nonlin = self.autoencoder.transform(x_mid)
        x_low = self.umap.transform(x_nonlin)
        x_low_rot_rescaled = self._rotate_and_scale(x_low)
        return x_low_rot_rescaled


class EgoEmbedder(object):

    def __init__(
            self,
            decomposition_funcs=None,
            preprocessors=None,
            nbits=12,
            seed=1,
            target_metric='categorical',
            n_components_mid=1024,
            n_components_low=32,
            n_components=2,
            embed_n_neighbors=10,
            context=2,
            count=2,
            grammar_n_neighbors=None,
            n_iter=0):
        self.vectorize = curry(ego_vectorize)(
            decomposition_funcs=decomposition_funcs,
            preprocessors=preprocessors,
            nbits=nbits,
            seed=seed)
        self.n_components_mid = n_components_mid
        self.svd = TruncatedSVD(n_components=n_components_mid)
        self.minmaxscaler = MinMaxScaler()
        self.normalizer = Normalizer()
        self.n_components_low = n_components_low
        self.autoencoder = AutoEncoder(input_dim=n_components_mid,
                                       encoding_dim=n_components_low)

        self.target_metric = target_metric
        self.umap = UMAP(
            n_components=n_components,
            n_neighbors=embed_n_neighbors,
            target_metric=target_metric)
        self.neigh = NeighborhoodEgoGraphGrammar(
            decomposition_function=decomposition_funcs,
            context=context, count=count, n_neighbors=grammar_n_neighbors)
        self.n_iter = n_iter

    def _make_variants(self, graphs):
        neigh_graphs = [new_g for g in graphs
                        for new_g in self.neigh.neighbors(g)]
        neigh_graphs = remove_duplicates(neigh_graphs)
        out_graphs = neigh_graphs[:]
        for i in range(self.n_iter - 1):
            neigh_graphs = [new_g for g in neigh_graphs
                            for new_g in self.neigh.neighbors(g)]
            neigh_graphs = remove_duplicates(neigh_graphs)
            out_graphs += neigh_graphs[:]
        return out_graphs

    def fit(self, orig_graphs, original_targets):
        """fit."""
        if self.target_metric == 'categorical':
            targets = LabelEncoder().fit_transform(original_targets)
        else:
            targets = original_targets
        assert(len(orig_graphs) >= self.n_components_mid), 'Num instances (%d) cannot be smaller than n_components_mid (%d)' % (
            len(orig_graphs), self.n_components_mid)
        if self.n_iter == 0:
            y = list(targets)
            graphs = orig_graphs
        else:
            self.neigh.fit(orig_graphs)
            logger.debug('Grammar: %s' % self.neigh)
            semi_graphs = self._make_variants(orig_graphs)
            logger.debug('# generated graphs: %d' % len(semi_graphs))
            y = list(targets) + [-1] * len(semi_graphs)
            graphs = orig_graphs + semi_graphs
        y = np.array(y)
        # graph to high dim sparse
        x = self.vectorize(graphs)

        # high dim sparse to mid dim dense
        self.svd.fit(x)
        x_mid = self.svd.transform(x)
        x_mid = self.minmaxscaler.fit_transform(x_mid)
        x_mid = self.normalizer.fit_transform(x_mid)

        # mid dim to lower dim using non linear mapping
        x_nonlin = self.autoencoder.fit(x_mid).transform(x_mid)

        # lower dim dense to low dim
        y = np.array(y)
        self.umap.fit(x_nonlin, y)
        x_low = self.umap.transform(x_nonlin)

        # rotation
        u, s, vh = np.linalg.svd(x_low, full_matrices=True)
        self.rotation = vh.T
        x_low_rot = np.dot(x_low, self.rotation)

        # scale
        self.xmin = np.min(x_low_rot, axis=0).reshape(-1)
        self.xmax = np.max(x_low_rot, axis=0).reshape(-1)
        self.xlen = np.absolute(self.xmax - self.xmin).max()

        return self

    def _rotate_and_scale(self, x):
        x_low_rot = np.dot(x, self.rotation)
        x_low_rot_rescaled = (x_low_rot - self.xmin) / self.xlen
        return x_low_rot_rescaled

    def transform(self, graphs):
        """transform."""
        x = self.vectorize(graphs)
        x_mid = self.svd.transform(x)
        x_mid = self.minmaxscaler.transform(x_mid)
        x_mid = self.normalizer.transform(x_mid)
        x_nonlin = self.autoencoder.transform(x_mid)
        x_low = self.umap.transform(x_nonlin)
        x_low_rot_rescaled = self._rotate_and_scale(x_low)
        return x_low_rot_rescaled


class EDeNEmbedder(object):

    def __init__(
            self,
            complexity=3,
            n_bits=14,
            target_metric='categorical',
            n_components_mid=1024,
            n_components_low=32,
            n_components=2,
            embed_n_neighbors=10,
            core=2,
            context=2,
            count=2,
            grammar_n_neighbors=None,
            n_iter=0):
        self.complexity = complexity
        self.nbits = n_bits
        self.n_components_mid = n_components_mid
        self.svd = TruncatedSVD(n_components=n_components_mid)
        self.n_components_low = n_components_low
        self.autoencoder = AutoEncoder(input_dim=n_components_mid,
                                       encoding_dim=n_components_low)

        self.target_metric = target_metric
        self.umap = UMAP(
            n_components=n_components,
            n_neighbors=embed_n_neighbors,
            target_metric=target_metric)
        self.neigh = NeighborhoodGraphGrammar(
            core=core, context=context, count=count, n_neighbors=grammar_n_neighbors)
        self.n_iter = n_iter

    def _make_variants(self, graphs):
        neigh_graphs = [new_g for g in graphs
                        for new_g in self.neigh.make_all_neighbors(
                            g, include_original=False)]
        neigh_graphs = remove_duplicates(neigh_graphs)
        out_graphs = neigh_graphs[:]
        for i in range(self.n_iter - 1):
            neigh_graphs = [new_g for g in neigh_graphs
                            for new_g in self.neigh.make_all_neighbors(
                                g, include_original=False)]
            neigh_graphs = remove_duplicates(neigh_graphs)
            out_graphs += neigh_graphs[:]
        return out_graphs

    def fit(self, orig_graphs, original_targets):
        """fit."""
        assert(len(orig_graphs) >= self.n_components_mid), 'Num instances (%d) cannot be smaller than n_components_mid (%d)' % (
            len(orig_graphs), self.n_components_mid)

        targets = LabelEncoder().fit_transform(original_targets)
        if self.n_iter == 0:
            y = list(targets)
            graphs = orig_graphs
        else:
            self.neigh.fit(orig_graphs)
            logger.debug('Grammar: %s' % self.neigh)
            semi_graphs = self._make_variants(orig_graphs)
            logger.debug('# generated graphs: %d' % len(semi_graphs))
            y = list(targets) + [-1] * len(semi_graphs)
            graphs = orig_graphs + semi_graphs
        x = eden_vectorize(
            graphs, complexity=self.complexity, nbits=self.nbits)
        self.svd.fit(x)
        x_mid = self.svd.transform(x)

        # mid dim to lower dim using non linear mapping
        x_nonlin = self.autoencoder.fit(x_mid).transform(x_mid)

        # lower dim dense to low dim
        y = np.array(y)
        self.umap.fit(x_nonlin, y)
        x_low = self.umap.transform(x_nonlin)

        # rotation
        u, s, vh = np.linalg.svd(x_low, full_matrices=True)
        self.rotation = vh.T
        x_low_rot = np.dot(x_low, self.rotation)

        # scale
        self.xmin = np.min(x_low_rot, axis=0).reshape(-1)
        self.xmax = np.max(x_low_rot, axis=0).reshape(-1)
        self.xlen = np.absolute(self.xmax - self.xmin).max()

        return self

    def _rotate_and_scale(self, x):
        x_low_rot = np.dot(x, self.rotation)
        x_low_rot_rescaled = (x_low_rot - self.xmin) / self.xlen
        return x_low_rot_rescaled

    def transform(self, graphs):
        """transform."""
        x = eden_vectorize(
            graphs, complexity=self.complexity, nbits=self.nbits)
        x_mid = self.svd.transform(x)
        x_nonlin = self.autoencoder.transform(x_mid)
        x_low = self.umap.transform(x_nonlin)
        x_low_rot_rescaled = self._rotate_and_scale(x_low)
        return x_low_rot_rescaled
