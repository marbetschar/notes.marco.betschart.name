import numpy as np
import matplotlib.pyplot as plt

def mvn_pdf(x, mu, Sigma, log=True):
    """ Returns the density of a multivariate Gaussian distribution
    with mean mu and covariace Sigma at point x

    Arguments:
    x     -- (Dx1) evaluation point
    mu    -- (Dx1) mean vector
    Sigma -- (DxD) covariance matrix
    log   -- (bool) if true, return log density. If false, return the density. (default=True)

    Returns:
    (scalar) density
    """
    D = len(mu)
    Sigma = Sigma + 1e-10 * np.identity(D)
    L = np.linalg.cholesky(Sigma)
    v = np.linalg.solve(L, x-mu)
    const_term = -0.5 * D * np.log(2 * np.pi)
    det_term = -0.5 * 2 * np.sum(np.log(np.diag(L)))
    quad_term = -0.5 * np.sum(v ** 2)
    if log:
        return const_term + det_term + quad_term
    else:
        return np.exp(const_term + det_term + quad_term)

def predict(x, a, b):
    """ returns prediction for inputs x given parameter (a,b)

    Arguments:
    x    --  (Nx1) vector of inputs
    a    --  slope parameter
    b    --  intercept parameter

    Returns:
    (Nx1) vector of predictions
    """
    return a*x + b


def log_prior(a, b, m, S):
    """ returns the log density of the prior at the points (a,b) given m and S

    Arguments:
    a    -- (scalar) slope parameter
    b    -- (scalar) intercept parameter
    m    -- (2x1) The prior mean
    S    -- (2x2) The prior covariance

    Returns
    (scalar) log density for the pair (a,b)

    """
    return mvn_pdf(np.array([a,b])[:, None], m, S, log=True)

def log_likelihood(x, y, a, b, sigma2):
    """ returns the log-likelihood for the data (x,y) given the parameters (a,b)

    Arguments:
    x    -- (Nx1) vector of inputs
    y    -- (Nx1) vector of responses
    a    -- slope parameter
    b    -- intercept parameter

    Returns:
    (scalar) log likelihood of (x,y)
    """
    log_npdf = lambda x, m, v: -0.5 * np.log(2* np.pi * v) - 0.5 * (x - m)**2/v
    return np.sum(log_npdf(y, predict(x,a,b), sigma2))


def log_posterior(x, y, a, b, m, S, sigma2):
    """Returns the log posterior at (a,b), given the data (x,y) and the prior parameters (m, S).

    Arguments:
    x    -- (Nx1) vector of inputs
    y    -- (Nx1) vector of responses
    a    -- slope parameter
    b    -- intercept parameter
    m    -- (2x1) prior mean of (a, b)
    S    -- (2x2) prior covariance of (a, b)
    sigma2 -- (scalar) noise variance

    Returns:
    (scalar) log posterior at (a, b)
    """
    return log_prior(a, b, m, S) + log_likelihood(x, y, a, b, sigma2)

def compute_posterior(x, y, m, S, sigma2):
    """ return the posterior mean and covariance of w given (x,y)
    and hyperparameters m, S and sigma2

    Arguments:
    x      -- (Nx1) vector of inputs
    y      -- (Nx1) vector of responses
    m      -- (Dx1) prior mean
    S      -- (DxD) prior covariance
    sigma2 -- (scalar) noise variance

    Returns:
    mu     -- (Dx1) posterior mean
    Sigma  -- (DxD) posterior covariance

    """
    import scipy as sc
    Sinv = np.linalg.inv(S)
    X = design_matrix(x)
    Sigmainv = Sinv + np.dot(X.T, X)/sigma2
    L = sc.linalg.cho_factor(Sigmainv)
    scaled_mu = np.divide(np.dot(X.T, y), sigma2).reshape(-1, 1) + np.linalg.solve(S, m)
    mu = sc.linalg.cho_solve(L, scaled_mu)
    Sigma = sc.linalg.cho_solve(L, np.identity(len(m)))

    assert mu.shape == m.shape
    assert S.shape == Sigma.shape

    return mu, Sigma

def design_matrix(x):
    """ returns the design matrix for a vector of input values x

    Arguments:
    x    -- (Nx1) vector of inputs

    Returns:
    (Nx2) design matrix

    """
    X = np.column_stack((x, np.ones(len(x))))
    return X

def generate_mvn_samples(mu, Sigma, M):
    """ return samples from a multivariate normal distribution N(mu, Sigma)

    Arguments:
    mu      -- (Dx1) mean vector
    Sigma   -- (DxD) covariance matrix
    M       -- (scalar) number of samples

    Returns:
    (DxM) matrix, where each column corresponds to a sample
    """

    jitter = 1e-8
    D = len(mu)
    L = np.linalg.cholesky(Sigma + jitter*np.identity(D))
    zs = np.random.normal(0, 1, size=(D, M))
    fs = mu + np.dot(L, zs)
    return fs

def compute_f_posterior(x, mu, Sigma):
    """ compute the posterior distribution of f(x) wrt. posterior distribution N(mu, Sigma)

    Arguments:
    x      -- (Nx1) vector of inputs
    mu     -- (2x1) mean vector
    Sigma  -- (2x2) covariance matrix

    Returns:
    mu_f   -- (Nx1) vector of pointwise posterior means at x
    var_f  -- (Nx1) vector of pointwise posterior variances at x

    """

    X = np.column_stack((x, np.ones(len(x))))
    mu_f = np.dot(X, mu)
    var_f = np.diag(np.dot(np.dot(X, Sigma), X.T))[:, None]

    return mu_f, var_f

def plot_prior_density(m, S, a, b):
    A_array, B_array = np.meshgrid(a, b)
    Z = np.array([log_prior(ai, bi, m, S) for (ai, bi) in zip(A_array.ravel(), B_array.ravel())])
    Z = Z.reshape((len(a), len(b)))
    plt.contour(a, b, log_to_density(Z), cmap='plasma')
    plt.xlabel('slope')
    plt.ylabel('intercept')
    plt.gca().set_aspect('equal', adjustable='box')

def plot_likelihood(x, y, sigma2, a, b):
    A_array, B_array = np.meshgrid(a, b)
    Z = np.array([log_likelihood(x, y, ai, bi, sigma2) for (ai, bi) in zip(A_array.ravel(), B_array.ravel())])
    Z = Z.reshape((len(a), len(b)))
    plt.contour(a, b, log_to_density(Z), 10, cmap='plasma')
    plt.xlabel('slope')
    plt.ylabel('intercept')
    plt.gca().set_aspect('equal', adjustable='box')

def plot_posterior_density(x, y, m, S, sigma2, a, b):
    A_array, B_array = np.meshgrid(a, b)
    Z = np.array([log_posterior(x, y, ai, bi, m, S, sigma2) for (ai, bi) in zip(A_array.ravel(), B_array.ravel())])
    Z = Z.reshape((len(a), len(b)))
    plt.contour(a, b, log_to_density(Z), 10, cmap='plasma')
    plt.xlabel('slope')
    plt.ylabel('intercept')
    plt.gca().set_aspect('equal', adjustable='box')

def log_to_density(Z):
    Z = Z - np.max(Z)
    Z = np.exp(Z)
    return Z/np.sum(Z)