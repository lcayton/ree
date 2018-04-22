import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize, scipy.spatial
import warnings

def cmds(D):
    n, _ = D.shape
    H = np.eye(n) - np.ones((n, n))/n
    B = -H.dot(D**2).dot(H)/2
    X = _gram_to_config(B)
    return X

def ree_error(D, Dp):
    # Returns the sum of the abolute elementwise
    # differences of D and Dp. Note: assumes that D
    # and Dp and both *squared* distance matrices.
    return np.sum(np.abs(D - Dp))

def _ree_grad(B, Dsq):
    # Computes the REE subgradient for a gram matrix B,
    # where Dsq is the (squared) EDM passed to REE.

    Dtemp = gram_dist(B)
    G = (Dtemp < Dsq) * 2 - 1  # sets off-diag entries of subgrad correctly
    G -= np.diag(np.diag(G))  # clear the diagonal entries
    g = np.sum(G, axis=1) * -1  # compute the diag entries
    G += np.diag(g)  # .. and set them
    return G


def ree(D, max_num_its=100000, no_line_search=False, init_zero=True, verbose=True):
    n, _ = D.shape
    D = 0.5 * (D + D.T)
    Dsq = D * D

    # init B as the Gram matrix of a random n-by-2 configuration.
    B = np.zeros_like(Dsq)
    if not init_zero:
        X = np.random.randn(n, 2) / np.sqrt(2)
        X = X - np.sum(X, 0) / X.shape[0]
        B = X.dot(X.T)

    it = 0
    not_done = True
    c = 0.1# np.float32(n)
    best_err = np.inf
    while not_done:
        def _error(b):
            B = _vec_to_sq_matrix(b)
            return ree_error(gram_dist(B), Dsq)

        def _grad(b):
            Bt = _vec_to_sq_matrix(b)
            Gt = _ree_grad(Bt, Dsq)
            return Gt.reshape(-1)

        G = - _grad(B.reshape(-1))

        alpha = None
        if not no_line_search:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The line search algorithm did not converge')
                (alpha, fc, gc, new_fval, old_fval, new_slope) = scipy.optimize.line_search(_error, _grad, B.reshape(-1), G)

        if not alpha:
            alpha = c / np.sqrt(it+1)
            # Alternative rule: gamma / norm(G)
            # alpha = 3 / np.linalg.norm(G)

        Bnew = B + alpha * _vec_to_sq_matrix(G)

        def _project(C):
            C = C - np.sum(C)/np.float(C.size)
            X = _gram_to_config(C)
            return X.dot(X.T)

        Bnew = _project(Bnew)
        B = Bnew
        new_err = ree_error(gram_dist(B), Dsq)
        best_err = np.min([best_err, new_err])
        if verbose and not it%10:
            print "it {}; error {:.2f} (best={:.2f}) [alpha={}]".format(it, new_err, best_err, alpha)
        it += 1
        not_done &= (it < max_num_its)

    X = _gram_to_config(B)
    return X

def _gram_to_config(B):
    evals, evecs = np.linalg.eigh(B)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    U = evecs[:,w]
    X = U.dot(L)
    return X

def _vec_to_sq_matrix(b):
    nsq = b.shape[0]
    n = int(np.sqrt(nsq))
    return b.reshape(n, n)

def gram_dist(B):
    n, _ = B.shape
    # Returns matrix D s.t. D_ij = B_ii + B_jj - B_ij - B_ji
    b = np.diag(B).reshape(-1, 1)  # make column vector out of B_ii's
    o = np.ones((n,1))
    return b.dot(o.T) + o.dot(b.T) - B - B.T


def plot(X, names):
    fig, ax = plt.subplots()
    ax.plot(X[:, 0], X[:, 1], 'o')
    for i, n in enumerate(names):
        ax.annotate(n, (X[i, 0], X[i, 1]))
    plt.show()

def dists(X):
    d = scipy.spatial.distance.pdist(X)
    return scipy.spatial.distance.squareform(d)

if __name__ == '__main__':
    D, names = cities_data()
    X = cmds(D)
    print('error = {}'.format(np.linalg.norm(D-dists(X))))
    plot(X, names)

