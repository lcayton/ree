# Robust Euclidean Embedding
This package is a python implementation of Robust Euclidean Embedding (REE). REE is an algorithm that takes as input a set of _n_ interpoint dissimilarities (distances) and tries to find _n_ points whose Euclidean distances match the input. This basic task is known as an embedding or multidimensional scaling. The algorithm is described in this paper:

L. Cayton and S. Dasgupta. *Robust Euclidean embedding*. _Twenty-Third International Conference on Machine Learning (ICML)_, 2006. 

This implementation is based on the _subgradient_ algorithm described in that paper. It also has an implementation of classical multidimensional scaling (cMDS).

Note that this implementation is *not* what was used to produce the results in the original paper; that was a matlab implementation that is long gone. It differs in at least one respect: in default usage, the algorithm will attempt to set the step size via a line search on each iteration. If the line search fails, it will fall back to using a step size of `c / sqrt(i)`, where _i_ is the iteration number.

### Usage
Basic usage.

```
import ree

...
# D is a n-by-n numpy array
X = ree(D)
# X will be a n-by-n numpy array. The rows correspond to points; the columns to their attributes. 
# To get a d-dimensional embedding, simply do:
X_lowdim = X[:, :d]

```

Options:

* Set a limit on the number of iterations:

```
X = ree(D, max_num_its=1000)
```
* Turn off the line search (always use a step size of `c/sqrt(n)`):

```
X = ree(D, no_line_search=True)
```
* Turn off verbosity:

```
X = ree(D, verbosity=False)
```
* Initialize solution to the zeros-matrix (otherwise a random n-by-2 dimensional configuration will be used):

```
X = ree(D, init_zeros=True)
```

### Requirements and limitations
This is currently only tested under Python 2.7. It requires only scipy and numpy.


