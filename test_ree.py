from ree import cmds, dists, gram_dist, ree
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

SEED = 297

@pytest.fixture
def random_state():
    return np.random.RandomState(SEED)


def random_config(n, d, random_state):
    return random_state.rand(n, d)

def test_gram_dist(random_state):
    X = random_config(20, 3, random_state)
    B = X.dot(X.T)

    D = dists(X)  # this is a wrapper around external code, so is assumed to be correct
    Dg = gram_dist(B)

    assert_almost_equal(D*D,  Dg)


def test_cmds(random_state):
    X = random_config(30, 4, random_state)
    D = dists(X)
    Xmds = cmds(D)
    assert np.linalg.matrix_rank(Xmds.dot(Xmds.T)) == 4
    assert_almost_equal(dists(Xmds), D)

def test_ree_edm(random_state):
    # See if REE can recover an actual euclidean matrix
    # with low error.
    X = random_config(10, 4, random_state)
    D = dists(X)
    Xree = ree(D, max_num_its=1000)
    assert np.sum(np.abs(dists(Xree) - D)) / np.sum(np.abs(D)) < .05
