import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn.manifold

def cities_data():
    # data from http://polisci.msu.edu/jacoby/iu/mds2012/citydists/citydists.txt
    # and symmetrized

    D = np.loadtxt('cities.txt')
    D_clean = 0.5 * (D + D.T)
    names = ('atl', 'chi', 'den', 'hou', 'LA', 'mia', 'NYC', 'SF', 'sea', 'WDC')
    return D_clean, names

def sklearn_mds(D):
    mds = sklearn.manifold.MDS(dissimilarity='precomputed')
    model = mds.fit(D)
    print model.stress_
    return model.embedding_

def cmds(D):
     n, _ = D.shape
     H = np.eye(n) - np.ones((n, n))/n
     B = -H.dot(D**2).dot(H)/2
     evals, evecs = np.linalg.eigh(B)

     idx   = np.argsort(evals)[::-1]
     evals = evals[idx]
     evecs = evecs[:,idx]

     w, = np.where(evals > 0)
     L = np.diag(np.sqrt(evals[w]))
     U = evecs[:,w]
     X = U.dot(L)
     return X

def ree(D, max_num_its=100, X0 = None):
    n, _ = D.shape
    D = 0.5 * (D + D.T)
    Dsq = D * D

    # init B as the Gram matrix of a random n-by-2 configuration.
    X = np.random.randn(n, 2) / np.sqrt(2)
    B = X.dot(X.T)
    it = 0
    not_done = True
    c = 0.1
    while not_done:
        def f(b):
            B = _vec_to_sq_matrix(b)
            print np.sum(np.abs(gram_dist(B) - Dsq))
            return np.sum(np.abs(gram_dist(B) - Dsq))

        def grad(b):
            Bt = _vec_to_sq_matrix(b)
            Dtemp = gram_dist(Bt)
            Gt = (Dtemp < D) * 2 - 1  # sets off-diag entries of subgrad correctly
            Gt -= np.diag(np.diag(Gt))  # clear the diagonal entries
            g = np.sum(Gt, axis=1) * -1  # compute the diag entries
            Gt += np.diag(g)  # .. and set them
            return Gt.reshape(-1)

        G = - grad(B.reshape(-1))
        # alpha = c / np.sqrt(it + 1.)
        alpha, fc, gc, new_fval, old_fval, new_slope = scipy.optimize.line_search(f, grad, B.reshape(-1), G)
        if alpha:
            Bnew = B + alpha * _vec_to_sq_matrix(G)
        else:
            Bnew = B
            not_done = False

        evals, evecs = np.linalg.eigh(Bnew)
        idx   = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]
        w, = np.where(evals > 0)
        L = np.diag(evals[w])
        U = evecs[:,w]
        Bnew = U.dot(L).dot(U.T)

        B = Bnew
        new_err = np.sum(np.abs(gram_dist(B) - Dsq))
        print "it {}; error {}".format(it, new_err)
        it += 1
        not_done &= (it < max_num_its)

    return U.dot(np.sqrt(L))

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
    import sklearn.metrics
    return sklearn.metrics.pairwise.euclidean_distances(X)

if __name__ == '__main__':
    D, names = cities_data()
    X = cmds(D)
    print('error = {}'.format(np.linalg.norm(D-dists(X))))
    plot(X, names)

