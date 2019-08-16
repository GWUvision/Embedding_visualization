from __future__ import division
from sklearn import manifold
import warnings
from time import time
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import _utils
from sklearn.manifold import _barnes_hut_tsne
from sklearn.externals.six import string_types
from sklearn.utils import deprecated
MACHINE_EPSILON = np.finfo(np.double).eps

def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.
    Parameters
    ----------
    distances : array, shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, None, desired_perplexity, verbose)
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P

def _joint_probabilities_nn(distances, neighbors, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.
    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).
    Parameters
    ----------
    distances : array, shape (n_samples, k)
        Distances of samples to its k nearest neighbors.
    neighbors : array, shape (n_samples, k)
        Indices of the k nearest-neighbors for each samples.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : csr sparse matrix, shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors.
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    n_samples, k = neighbors.shape
    distances = distances.astype(np.float32, copy=False)
    neighbors = neighbors.astype(np.int64, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, neighbors, desired_perplexity, verbose)
    assert np.all(np.isfinite(conditional_P)), \
        "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix((conditional_P.ravel(), neighbors.ravel(),
                    range(0, n_samples * k + 1, k)),
                   shape=(n_samples, n_samples))
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s"
              .format(duration))
    return P

def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components,
                   skip_num_points=0, compute_error=True):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.
    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    compute_error: bool (optional, default:True)
        If False, the kl_divergence is not computed and returns NaN.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(
            P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad

def _kl_divergence_bh(params, P, degrees_of_freedom, n_samples, n_components,
                      angle=0.5, skip_num_points=0, verbose=False,
                      compute_error=True):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.
    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2)
    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.
    P : csr sparse matrix, shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    angle : float (default: 0.5)
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    verbose : int
        Verbosity level.
    compute_error: bool (optional, default:True)
        If False, the kl_divergence is not computed and returns NaN.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                      grad, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad

def _kl_divergence_yoke(params_X,params_Y, P_X,P_Y, degrees_of_freedom, n_samples, n_components,alpha=0,fixed_Y=False,y=None,
                   skip_num_points=0, compute_error=True,oneplot=False):
    """yoke t-SNE objective function: gradient of the KL divergence
    of two p_ijs and q_ijs and the absolute error, also with a L2 distance to align two maps.
    """
    X_embedded = params_X.reshape(n_samples, n_components)
    Y_embedded = params_Y.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist_X = pdist(X_embedded, "sqeuclidean")
    dist_X /= degrees_of_freedom
    dist_X += 1.
    dist_X **= (degrees_of_freedom + 1.0) / -2.0
    Q_X = np.maximum(dist_X / (2.0 * np.sum(dist_X)), MACHINE_EPSILON)

    if not fixed_Y:
        dist_Y = pdist(Y_embedded, "sqeuclidean")
        dist_Y /= degrees_of_freedom
        dist_Y += 1.
        dist_Y **= (degrees_of_freedom + 1.0) / -2.0
        Q_Y = np.maximum(dist_Y / (2.0 * np.sum(dist_Y)), MACHINE_EPSILON)
    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence_X = 2.0 * np.dot(
            P_X, np.log(np.maximum(P_X, MACHINE_EPSILON) / Q_X))
        if not fixed_Y:
            kl_divergence_Y = 2.0 * np.dot(
                P_Y, np.log(np.maximum(P_Y, MACHINE_EPSILON) / Q_Y))
    else:
        kl_divergence_X = np.nan
        kl_divergence_Y = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad_X = np.zeros(X_embedded.shape, dtype=np.float32)
    grad_Y = np.zeros(Y_embedded.shape, dtype=np.float32)
    PQd_X = squareform((P_X - Q_X) * dist_X)
    if not fixed_Y:
        PQd_Y = squareform((P_Y - Q_Y) * dist_Y)
        for i in range(0, n_samples):
            grad_Y[i] = np.dot(np.ravel(PQd_Y[i], order='K'),
                            Y_embedded[i] - Y_embedded)
    for i in range(0, n_samples):
        grad_X[i] = np.dot(np.ravel(PQd_X[i], order='K'),
                        X_embedded[i] - X_embedded)
    
    if y is None:
        grad_X += 2*alpha* (X_embedded-Y_embedded)
        if not fixed_Y:
            grad_Y += 2*alpha* (Y_embedded-X_embedded)
    
    if y is not None:
        cluster_grad_X = np.zeros(grad_X.shape)
        cluster_grad_Y = np.zeros(grad_Y.shape)
        dst = {}
        gradient = {}
        for i in set(y):
            index = np.where(y==i)
            Xcenter = X_embedded[index].sum(axis=0)/len(index[0])
            Ycenter = Y_embedded[index].sum(axis=0)/len(index[0])
            dst[i]=(distance.euclidean(Xcenter,Ycenter))
            gradient[i]=(Xcenter-Ycenter)/len(index[0])
        threshold = np.percentile(np.asarray(list(dst.values())),ratio*100)
        for key in dst.keys():
            if dst[key]>threshold:
                dst[key]=0
                gradient[key]=0
        for i in set(y):
            index = np.where(y==i)
            cluster_grad_X[index] = gradient[i]
            if not fixed_Y:
                cluster_grad_Y[index] = -gradient[i]
        grad_X = grad_X + 2*alpha*cluster_grad_X    
        grad_Y = grad_Y + 2*alpha*cluster_grad_Y    
    grad_X = grad_X.ravel()
    grad_Y = grad_Y.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad_X *= c
    grad_Y *= c
    
    error = kl_divergence_X+kl_divergence_Y+alpha*np.power((X_embedded-Y_embedded),2).sum()

    return error, errorx, errory, grad_X, grad_Y


def _kl_divergence_yoke_bh(params_X,params_Y, P_X,P_Y, degrees_of_freedom, n_samples, n_components,alpha=0,
                      angle=0.5, skip_num_points=0, verbose=False,fixed_Y=False,y=None,
                      compute_error=True,oneplot=False,ratio=1.0):
    """yoke t-sne by bhtree. Adding a L2 distance than def _kl_divergence_bh
    """
    params_X = params_X.astype(np.float32, copy=False)
    params_Y = params_Y.astype(np.float32, copy=False)
    X_embedded = params_X.reshape(n_samples, n_components)
    Y_embedded = params_Y.reshape(n_samples, n_components)

    val_P_X = P_X.data.astype(np.float32, copy=False)
    neighbors_X = P_X.indices.astype(np.int64, copy=False)
    indptr_X = P_X.indptr.astype(np.int64, copy=False)
    
    if not fixed_Y:
        val_P_Y = P_Y.data.astype(np.float32, copy=False)
        neighbors_Y = P_Y.indices.astype(np.int64, copy=False)
        indptr_Y = P_Y.indptr.astype(np.int64, copy=False)
        
        

    grad_X = np.zeros(X_embedded.shape, dtype=np.float32)
    
    grad_Y = np.zeros(Y_embedded.shape, dtype=np.float32)
    
    
    errorx = _barnes_hut_tsne.gradient(val_P_X, X_embedded, neighbors_X, indptr_X,
                                      grad_X, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error)
    
        
    errory = 0
    
    if not fixed_Y and not oneplot:
        errory = _barnes_hut_tsne.gradient(val_P_Y, Y_embedded, neighbors_Y, indptr_Y,
                                      grad_Y, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error)
    if oneplot and not fixed_Y:
        errory = _barnes_hut_tsne.gradient(val_P_Y, X_embedded, neighbors_Y, indptr_Y,
                                      grad_Y, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error)
        grad_X += grad_Y
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    
    if not oneplot:
        if y is None:
            grad_X += 2*alpha* (X_embedded-Y_embedded)

            if not fixed_Y:
                grad_Y += 2*alpha* (Y_embedded-X_embedded)
    
        if y is not None:
            cluster_grad_X = np.zeros(grad_X.shape)
            cluster_grad_Y = np.zeros(grad_Y.shape)
            dst = {}
            gradient = {}
            for i in set(y):
                index = np.where(y==i)
                Xcenter = X_embedded[index].sum(axis=0)/len(index[0])
                Ycenter = Y_embedded[index].sum(axis=0)/len(index[0])
                dst[i]=(distance.euclidean(Xcenter,Ycenter))
                gradient[i]=(Xcenter-Ycenter)/len(index[0])
            threshold = np.percentile(np.asarray(list(dst.values())),ratio*100)
            for key in dst.keys():
                if dst[key]>threshold:
                    dst[key]=0
                    gradient[key]=0
            for i in set(y):
                index = np.where(y==i)
                cluster_grad_X[index] = gradient[i]
                if not fixed_Y:
                    cluster_grad_Y[index] = -gradient[i]
            grad_X = grad_X + 2*alpha*cluster_grad_X    
            grad_Y = grad_Y + 2*alpha*cluster_grad_Y 
    
    grad_X = grad_X.ravel()
    grad_X *= c

    grad_Y = grad_Y.ravel()
    grad_Y *= c
    
    error = errorx+errory+alpha*np.power((X_embedded-Y_embedded),2).sum()
    
    return error, errorx, errory, grad_X, grad_Y





def _gradient_descent_yoked(objective, X,Y, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None,fixed_Y=False,y=None,alpha=0, kwargs=None):
    """Batch gradient descent with momentum and individual gains in two t-sne processing.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    px = X.copy().ravel()
    py = Y.copy().ravel()
    updatex = np.zeros_like(px)
    gainsx = np.ones_like(px)
    updatey = np.zeros_like(py)
    gainsy = np.ones_like(py)
    
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, errorx, errory, gradx, grady = objective(px,py, *args, **kwargs)
        gradx_norm = linalg.norm(gradx)
        grady_norm = linalg.norm(grady)
        
        
        incx = updatex * gradx < 0.0
        decx = np.invert(incx)
        gainsx[incx] += 0.2
        gainsx[decx] *= 0.8
        
        incy = updatey * grady < 0.0
        decy = np.invert(incy)
        gainsy[incy] += 0.2
        gainsy[decy] *= 0.8
        
        
        np.clip(gainsx, min_gain, np.inf, out=gainsx)
        gradx *= gainsx
        updatex = momentum * updatex - learning_rate * gradx
        px += updatex
        
        np.clip(gainsy, min_gain, np.inf, out=gainsy)
        grady *= gainsy
        updatey = momentum * updatey - learning_rate * grady
        py += updatey

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if gradx_norm <= min_grad_norm and grady_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return px,py, error,errorx,errory, i


def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
    """Batch gradient descent with momentum and individual gains.
    Parameters
    ----------
    objective : function or callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.
    p0 : array-like, shape (n_params,)
        Initial parameter vector.
    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).
    n_iter : int
        Maximum number of gradient descent iterations.
    n_iter_check : int
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.
    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization.
    momentum : float, within (0.0, 1.0), optional (default: 0.8)
        The momentum generates a weight for previous gradients that decays
        exponentially.
    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.
    min_gain : float, optional (default: 0.01)
        Minimum individual gain for each parameter.
    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be aborted.
    verbose : int, optional (default: 0)
        Verbosity level.
    args : sequence
        Arguments to pass to objective function.
    kwargs : dict
        Keyword arguments to pass to objective function.
    Returns
    -------
    p : array, shape (n_params,)
        Optimum parameters.
    error : float
        Optimum.
    i : int
        Last iteration.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i


def trustworthiness(X, X_embedded, n_neighbors=5,
                    precomputed=False, metric='euclidean'):
    r"""Expresses to what extent the local structure is retained.
    The trustworthiness is within [0, 1]. It is defined as
    .. math::
        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))
    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.
    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten
    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.
    X_embedded : array, shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.
    n_neighbors : int, optional (default: 5)
        Number of neighbors k that will be considered.
    precomputed : bool, optional (default: False)
        Set this flag if X is a precomputed square distance matrix.
        ..deprecated:: 0.20
            ``precomputed`` has been deprecated in version 0.20 and will be
            removed in version 0.22. Use ``metric`` instead.
    metric : string, or callable, optional, default 'euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, see the
        documentation of argument metric in sklearn.pairwise.pairwise_distances
        for a list of available metrics.
    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    if precomputed:
        warnings.warn("The flag 'precomputed' has been deprecated in version "
                      "0.20 and will be removed in 0.22. See 'metric' "
                      "parameter instead.", DeprecationWarning)
        metric = 'precomputed'
    dist_X = pairwise_distances(X, metric=metric)
    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = NearestNeighbors(n_neighbors).fit(X_embedded).kneighbors(
        return_distance=False)

    n_samples = X.shape[0]
    t = 0.0
    ranks = np.zeros(n_neighbors)
    for i in range(n_samples):
        for j in range(n_neighbors):
            ranks[j] = np.where(ind_X[i] == ind_X_embedded[i, j])[0][0]
        ranks -= n_neighbors
        t += np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t


class Yoked_TSNE(BaseEstimator):
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean",init_ratio=1,init="random", init_X="random", init_Y="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init_ratio=1
        self.init = init
        self.init_X = init_X
        self.init_Y = init_Y
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle

    def _fit(self, X, skip_num_points=0):
        """Fit the model using X as training data.
        Note that sparse arrays can only be handled by method='exact'.
        It is recommended that you convert your sparse array to dense
        (e.g. `X.toarray()`) if it fits in memory, or otherwise using a
        dimensionality reduction technique (e.g. TruncatedSVD).
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. Note that this
            when method='barnes_hut', X cannot be a sparse array and if need be
            will be converted to a 32 bit float array. Method='exact' allows
            sparse arrays and 64bit floating point inputs.
        skip_num_points : int (optional, default:0)
            This does not compute the gradient for points with indices below
            `skip_num_points`. This is useful when computing transforms of new
            data where you'd like to keep the old data fixed.
        """
        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.metric == "precomputed":
            if isinstance(self.init, string_types) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be "
                                 "used with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")
            if np.any(X < 0):
                raise ValueError("All distances should be positive, the "
                                 "precomputed distances given as X is not "
                                 "correct")
        if self.method == 'barnes_hut' and sp.issparse(X):
            raise TypeError('A sparse matrix was passed, but dense '
                            'data is required for method="barnes_hut". Use '
                            'X.toarray() to convert to a dense numpy array if '
                            'the array is small enough for it to fit in '
                            'memory. Otherwise consider dimensionality '
                            'reduction techniques (e.g. TruncatedSVD)')
        if self.method == 'barnes_hut':
            X = check_array(X, ensure_min_samples=2,
                            dtype=[np.float32, np.float64])
        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                            dtype=[np.float32, np.float64])
        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the "
                             "barnes_hut algorithm as it relies on "
                             "quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is {}"
                             .format(self.early_exaggeration))

        if self.n_iter < 250:
            raise ValueError("n_iter should be at least 250")

        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    distances = pairwise_distances(X, metric=self.metric,
                                                   squared=True)
                else:
                    distances = pairwise_distances(X, metric=self.metric)

                if np.any(distances < 0):
                    raise ValueError("All distances should be positive, the "
                                     "metric given is not correct")

            # compute the joint probability distribution for the input space
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), ("All probabilities should be less "
                                    "or then equal to one")

        else:
            # Cpmpute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            k = min(n_samples - 1, int(3. * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors...".format(k))

            # Find the nearest neighbors for every point
            knn = NearestNeighbors(algorithm='auto', n_neighbors=k,
                                   metric=self.metric)
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                    n_samples, duration))

            t0 = time()
            distances_nn, neighbors_nn = knn.kneighbors(
                None, n_neighbors=k)
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Computed neighbors for {} samples in {:.3f}s..."
                      .format(n_samples, duration))

            # Free the memory used by the ball_tree
            del knn

            if self.metric == "euclidean":
                # knn return the euclidean distance but we need it squared
                # to be consistent with the 'exact' method. Note that the
                # the method was derived using the euclidean method as in the
                # input space. Not sure of the implication of using a different
                # metric.
                distances_nn **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities_nn(distances_nn, neighbors_nn,
                                        self.perplexity, self.verbose)

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = self.init_ratio*1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)
    

    def _yoke(self, X,Y,alpha,y=None,fixed_Y=False,oneplot=False,skip_num_points=0,ratio=1.0):
        """Fit the model using X and Y as training data. This method is trying to aligning two t-sne plot.
        """
        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.metric == "precomputed":
            if isinstance(self.init_X, string_types) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be "
                                 "used with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")
            if np.any(X < 0):
                raise ValueError("All distances should be positive, the "
                                 "precomputed distances given as X is not "
                                 "correct")
            if isinstance(self.init_Y, string_types) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be "
                                 "used with metric=\"precomputed\".")
            if fixed_Y==False and Y.shape[0] != Y.shape[1]:
                raise ValueError("X should be a square distance matrix")
            if fixed_Y==False and np.any(Y < 0):
                raise ValueError("All distances should be positive, the "
                                 "precomputed distances given as X is not "
                                 "correct")
        if (self.method == 'barnes_hut' and sp.issparse(X)) or (fixed_Y==False and self.method == 'barnes_hut' and sp.issparse(Y)):
            raise TypeError('A sparse matrix was passed, but dense '
                            'data is required for method="barnes_hut". Use '
                            'X.toarray() to convert to a dense numpy array if '
                            'the array is small enough for it to fit in '
                            'memory. Otherwise consider dimensionality '
                            'reduction techniques (e.g. TruncatedSVD)')
        if self.method == 'barnes_hut':
            X = check_array(X, ensure_min_samples=2,
                            dtype=[np.float32, np.float64])
            if not fixed_Y:
                Y = check_array(Y, ensure_min_samples=2,
                            dtype=[np.float32, np.float64])
        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                            dtype=[np.float32, np.float64])
            if not fixed_Y:
                Y = check_array(Y, accept_sparse=['csr', 'csc', 'coo'],
                            dtype=[np.float32, np.float64])
        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the "
                             "barnes_hut algorithm as it relies on "
                             "quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is {}"
                             .format(self.early_exaggeration))

        if self.n_iter < 250:
            raise ValueError("n_iter should be at least 250")

        n_samples = X.shape[0]

        neighbors_nn_X = None
        neighbors_nn_Y = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances_X = X
                distances_Y = Y
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    distances_X = pairwise_distances(X, metric=self.metric,
                                                   squared=True)
                    if not fixed_Y:
                        distances_Y = pairwise_distances(Y, metric=self.metric,
                                                   squared=True)
                else:
                    distances_X = pairwise_distances(X, metric=self.metric)
                    if not fixed_Y:
                        distances_Y = pairwise_distances(Y, metric=self.metric)

                if np.any(distances_X < 0):
                    raise ValueError("All distances should be positive, the "
                                     "metric given is not correct")
                if np.any(distances_Y<0):
                    raise ValueError("All distances should be positive, the "
                                     "metric given is not correct")

            # compute the joint probability distribution for the input space
            P_X = _joint_probabilities(distances_X, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P_X)), "All probabilities should be finite"
            assert np.all(P_X >= 0), "All probabilities should be non-negative"
            assert np.all(P_X <= 1), ("All probabilities should be less "
                                    "or then equal to one")
            if not fixed_Y:
                P_Y = _joint_probabilities(distances_Y, self.perplexity, self.verbose)
                assert np.all(np.isfinite(P_Y)), "All probabilities should be finite"
                assert np.all(P_Y >= 0), "All probabilities should be non-negative"
                assert np.all(P_Y <= 1), ("All probabilities should be less "
                                    "or then equal to one")
        

        else:
            # Cpmpute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            k = min(n_samples - 1, int(3. * self.perplexity + 1))
            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors...".format(k))

            # Find the nearest neighbors for every point
            knn1 = NearestNeighbors(algorithm='auto', n_neighbors=k,
                                   metric=self.metric)
            t0 = time()
            knn1.fit(X)
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                    n_samples, duration))

            t0 = time()
            distances_nn_X, neighbors_nn_X = knn1.kneighbors(
                None, n_neighbors=k)
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Computed neighbors for {} samples in {:.3f}s..."
                      .format(n_samples, duration))

            # Free the memory used by the ball_tree
            del knn1
            
            if not fixed_Y:
                knn2 = NearestNeighbors(algorithm='auto', n_neighbors=k,
                                   metric=self.metric)
                t0 = time()
                knn2.fit(Y)
                duration = time() - t0
                if self.verbose:
                    print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                        n_samples, duration))

                t0 = time()
                distances_nn_Y, neighbors_nn_Y = knn2.kneighbors(
                    None, n_neighbors=k)
                duration = time() - t0
                if self.verbose:
                    print("[t-SNE] Computed neighbors for {} samples in {:.3f}s..."
                          .format(n_samples, duration))

                # Free the memory used by the ball_tree
                del knn2
                
            if self.metric == "euclidean":
                # knn return the euclidean distance but we need it squared
                # to be consistent with the 'exact' method. Note that the
                # the method was derived using the euclidean method as in the
                # input space. Not sure of the implication of using a different
                # metric.
                distances_nn_X **= 2
                if not fixed_Y:
                    distances_nn_Y **= 2

            # compute the joint probability distribution for the input space
            P_X = _joint_probabilities_nn(distances_nn_X, neighbors_nn_X,
                                        self.perplexity, self.verbose)
            P_Y = 0
            if not fixed_Y:
                P_Y = _joint_probabilities_nn(distances_nn_Y, neighbors_nn_Y,
                                        self.perplexity, self.verbose)

        if isinstance(self.init_Y, np.ndarray):
            Y_embedded = self.init_Y
        elif self.init_Y == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            Y_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init_Y == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            Y_embedded = 1e-4 * random_state.randn(n_samples, self.n_components).astype(np.float32)
            if not fixed_Y:
                Y_embedded = 1e-4 * random_state.randn(n_samples, self.n_components).astype(np.float32)
        if isinstance(self.init_X,np.ndarray):
            X_embedded = self.init_X
        elif self.init_X == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            if not fixed_Y:
                Y_embedded = pca.fit_transform(Y).astype(np.float32, copy=False)
        elif self.init_X == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = self.init_ratio*1e-4 * random_state.randn(n_samples, self.n_components).astype(np.float32)
            if not fixed_Y:
                Y_embedded = self.init_ratio*1e-4 * random_state.randn(n_samples, self.n_components).astype(np.float32)
        elif not isinstance(self.init_Y, np.ndarray) and fixed_Y==True:
            raise ValueError("Must be given fixed Y")
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")
        
        if self.init_Y == 'same':
            Y_embedded = X_embedded

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)
        #print('same initialization:',X_embedded==Y_embedded)

        return self._yoke_tsne(P_X,P_Y, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          Y_embedded=Y_embedded,
                          neighbors_X=neighbors_nn_X,
                          neighbors_Y=neighbors_nn_Y,
                          skip_num_points=skip_num_points,y=y,alpha=alpha,fixed_Y=fixed_Y,oneplot=oneplot,ratio=ratio)


    @property
    @deprecated("Attribute n_iter_final was deprecated in version 0.19 and "
                "will be removed in 0.21. Use ``n_iter_`` instead")
    def n_iter_final(self):
        return self.n_iter_

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
              neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
        else:
            obj_func = _kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exageration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                      **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                          **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded
    

    def _yoke_tsne(self, P_X, P_Y, degrees_of_freedom, n_samples, X_embedded, Y_embedded,
              neighbors_X=None,neighbors_Y=None,y=None,fixed_Y = False, skip_num_points=0,alpha=0,oneplot=False,ratio=1.0):
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params_X = X_embedded.ravel()
        params_Y = Y_embedded.ravel()
        
        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P_X,P_Y, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        
        opt_args['kwargs']['alpha'] = alpha
        opt_args['kwargs']['fixed_Y'] = fixed_Y
        opt_args['kwargs']['y']=y
        opt_args['kwargs']['oneplot']=oneplot
        opt_args['kwargs']['ratio']=ratio
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_yoke_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
        else:
            obj_func = _kl_divergence_yoke

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exageration parameter
        P_X *= self.early_exaggeration
        if not fixed_Y:
            P_Y *= self.early_exaggeration
        
        params_X,params_Y, error,kl_x,kl_y, it = _gradient_descent_yoked(obj_func, params_X,params_Y,
                                                      **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P_X /= self.early_exaggeration
        if not fixed_Y:
            P_Y /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params_X,params_Y, error,kl_x,kl_y, it = _gradient_descent_yoked(obj_func, params_X,params_Y,
                                                      **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params_X.reshape(n_samples, self.n_components)
        Y_embedded = params_Y.reshape(n_samples, self.n_components)
        self.kl_divergencex = kl_x
        self.kl_divergencey = kl_y
        self.error = error

        return X_embedded,Y_embedded
    
    
    
    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        y : Ignored
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        y : Ignored
        """
        self.fit_transform(X)
        return self
    
    def Yoke_transform(self, X,Y,alpha,y=None, fixed_Y=False,oneplot=False,ratio=1.0):
        """Fit X Y into an embedded space and return that transformed
        output.
        """
        embedding_X,embedding_Y = self._yoke(X,Y,alpha,y=y,fixed_Y = fixed_Y,oneplot=oneplot,ratio=ratio)
        self.embedding_X = embedding_X
        self.embedding_Y = embedding_Y
        return self.embedding_X,self.embedding_Y

    def Yoke(self, X,Y, y=None,fixed_Y=False,oneplot=False,ratio=1.0):
        """Fit X Y into an embedded space.
        """
        self._yoke(X,Y,alpha,y=y,fixed_Y = fixed_Y)
        return self