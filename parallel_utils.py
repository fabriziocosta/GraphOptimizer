#!/usr/bin/env python
"""Provides the parallelization."""

import dill
from multiprocessing import Pool


def run_dill_encoded(what):
    """
    Use dill as replacement for pickle to enable multiprocessing on instance methods
    """
    fun, args = dill.loads(what)
    return fun(*args)


def _apply_async(pool, fun, args, callback):
    """
    Wrapper around apply_async() from multiprocessing, to use dill instead of pickle.
    This is a workaround to enable multiprocessing of classes.
    """
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),), callback=callback)


def parallel_map(func, iterable):
    res = []
    it_list = list(iterable)

    def my_int_callback(result):
        res.append(result)

    pool = Pool()
    for i, item in enumerate(it_list):
        _apply_async(pool, func, args=(item, i), callback=my_int_callback)
    pool.close()
    pool.join()
    results = [item for i, item in sorted(res, key=lambda x:x[0])]
    return results
