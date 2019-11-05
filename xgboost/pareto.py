import numpy as np
from cvxopt import matrix, solvers


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def compute_grad(G, alpha):
    if len(alpha.shape) == 1:
        np.expand_dims(alpha,1)
    return np.squeeze(2 * np.dot(np.dot(G.T, G),alpha))

def PGD(G, alpha0, eta=0.1, iter=500):
    # Running Projected Gradient Descent with simplex projection
    alpha0 = euclidean_proj_simplex(alpha0)
    alpha = np.zeros((iter,alpha0.shape[0]))
    alpha[0,:] = alpha0


    for i in range(1,iter):
        new_alpha = alpha[i-1,:] - (eta) * compute_grad(G,alpha[i-1,:])
        alpha[i,:] = euclidean_proj_simplex(new_alpha)
    
    return alpha[-1,:]

def solve_cvxopt(GradMat):
    d,c = GradMat.shape
    P = 2*matrix(np.dot(GradMat.T, GradMat))
    q = matrix(np.zeros(c))
    G = matrix(-1.0 * np.eye(c))
    h = matrix(np.zeros(c))
    A = matrix(np.ones(c), (1,c))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b, show_progress=False)
    return np.squeeze(np.array(sol['x']))

def normalization(grad_mat,losses=None, type='norm2'):

    if np.linalg.norm(grad_mat) == 0:
        return grad_mat

    if type=='norm2':
        return grad_mat / np.linalg.norm(grad_mat,2,axis=0)
    elif type=='norm1':
        return grad_mat / np.linalg.norm(grad_mat,1,axis=0)
    elif type=='loss':
        if losses.any():
            return grad_mat / losses
        else:
            raise ValueError('Losses should be passed for type "loss"')
    elif type=='loss_norm':
        if losses.any():
            return (grad_mat * losses) / np.linalg.norm(grad_mat,2,axis=0)**2
        else:
            raise ValueError('Losses should be passed for type "loss_norm"')
    else:
        raise NotImplementedError

def find_pareto_descent(grads,
                        normalization_type='norm1',
                        solver='CVX',
                        losses=None
                        ):
  norm_grad_mat = normalization(grads, np.array(losses), type=normalization_type) 
  if solver == 'CVX':
    alpha = solve_cvxopt(norm_grad_mat)
  elif solver == 'PGD':
    alpha = PGD(norm_grad_mat, np.random.rand(3), iter=1000, eta=0.1)
  else:
    raise ValueError('Losses should be passed for type "with_loss"')
  return np.dot(grads, np.expand_dims(alpha, 1)).T


if __name__=="__main__":
  grads = np.array([[1, 0],
                    [0, 1],
                    [0, 1]])
  grads_mat = grads

  pareto_descent = find_pareto_descent(grads_mat)
  print(pareto_descent)