from ree import cmds, dists, gram_dist, ree, ree_error
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

SEED = 297


@pytest.fixture
def random_state():
    return np.random.RandomState(SEED)


@pytest.fixture
def cities_data():
    # Standard data used to demonstrate MDS algorithms.
    # Derived from http://polisci.msu.edu/jacoby/iu/mds2012/citydists/citydists.txt.
    # Symmetrized.
    D = np.array(
        [[0, 0.587, 1.212, 0.701, 1.936, 0.604, 0.748, 2.139, 2.182, 0.543],
         [0.587, 0, 0.920, 0.940, 1.745, 1.188, 0.713, 1.858, 1.737, 0.597],
         [1.212, 0.920, 0, 0.879, 0.831, 1.726, 1.631, 0.949, 1.021, 1.494],
         [0.701, 0.940, 0.879, 0, 1.374, 0.968, 1.420, 1.645, 1.891, 1.220],
         [1.936, 1.745, 0.831, 1.374, 0, 2.339, 2.451, 0.347, 0.959, 2.300],
         [0.604, 1.188, 1.726, 0.968, 2.339, 0, 1.092, 2.594, 2.734, 0.923],
         [0.748, 0.713, 1.631, 1.420, 2.451, 1.092, 0, 2.571, 2.408, 0.205],
         [2.139, 1.858, 0.949, 1.645, 0.347, 2.594, 2.571, 0, 0.678, 2.442],
         [2.182, 1.737, 1.021, 1.891, 0.959, 2.734, 2.408, 0.678, 0, 2.329],
         [0.543, 0.597, 1.494, 1.229, 2.300, 0.923, 0.205, 2.442, 2.329, 0]])
    D_clean = 0.5 * (D + D.T)
    names = (
        'atl',
        'chi',
        'den',
        'hou',
        'LA',
        'mia',
        'NYC',
        'SF',
        'sea',
        'WDC')
    return D_clean, names


def random_config(n, d, random_state):
    return random_state.rand(n, d)


def rel_ree_error(Dfound, D):
    return ree_error(Dfound, D) / np.sum(np.abs(D))


def test_cities(cities_data):
    D, _ = cities_data
    X = cmds(D)
    assert rel_ree_error(dists(X), D) < .05

    Xree = ree(D, 100)
    assert rel_ree_error(dists(Xree), D) < .05


def test_gram_dist(random_state):
    X = random_config(20, 3, random_state)
    B = X.dot(X.T)
    D = dists(X)
    Dg = gram_dist(B)

    assert_almost_equal(D * D, Dg)


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
    assert rel_ree_error(dists(Xree), D) < .05


@pytest.fixture
def sparse_perturb(random_state):
    num_perturb = 2
    n = 10
    d = 3
    X = random_config(n, d, random_state)
    D = dists(X)
    eps = 100 * np.max(D)
    noise = eps * np.ones_like(D)
    noise = np.triu(noise, 1) + np.triu(noise, 1).T
    Dp = D.copy()

    for p in range(num_perturb):
        i = random_state.randint(1, n)
        j = random_state.randint(i + 1, n + 1)
        Dp[i, j] += noise[i, j]
        Dp[j, i] += noise[j, i]

    return D, Dp


def test_ree_sparse_perturbed(sparse_perturb):
    D, Dp = sparse_perturb

    Xree = ree(Dp, max_num_its=1000)
    Dree = dists(Xree)
    assert rel_ree_error(dists(Xree), D) < 0.05
