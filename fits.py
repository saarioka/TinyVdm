import numpy as np
import numba as nb
from scipy.special import gamma


@nb.njit()
def sgconst(x, peak, mean, capsigma, const):
    return const + peak*np.exp(-(x-mean)**2/(2*capsigma**2))


@nb.njit()
def sg(x, peak, mean, capsigma):
    return sgconst(x, peak, mean, capsigma, 0)


@nb.njit()
def dgconst(x, peak, mean, capsigma, frac, capsigma_ratio, const):
    c = capsigma / (frac * capsigma_ratio + 1 - frac)
    return const + peak * (sg(x, frac, mean, capsigma_ratio*c) + sg(x, 1-frac, mean, c))


@nb.njit()
def dg(x, peak, mean, capsigma, frac, capsigma_ratio):
    return dgconst(x, peak, mean, capsigma, frac, capsigma_ratio, 0)


def supergconst(x, peak, mean, capsigma, p, const):
    beta = p*2.0
    alpha = np.sqrt(2*np.pi)*beta / (2*gamma(1.0/beta))*capsigma
    return const + peak*np.exp(-(np.abs(x-mean)/alpha)**beta)


def superg(x, peak, mean, capsigma, p):
    return supergconst(x, peak, mean, capsigma, p, 0)


@nb.njit()
def polyg6const(x, peak, mean, capsigma, r2, r4, r6, const):
    x0 = x-mean
    sigma = capsigma / (1 + r2 + 3*r4 + 15*r6)
    return const + peak * (1 + r2*(x0/sigma)**2 + r4*(x0/sigma)**4 + r6*(x0/sigma)**6) * np.exp(-(x0/sigma)**2/2)


@nb.njit()
def polyg6(x, peak, mean, capsigma, r2, r4, r6):
    return polyg6const(x, peak, mean, capsigma, r2, r4, r6, 0)


@nb.njit()
def polyg4(x, peak, mean, capsigma, r2, r4):
    return polyg6const(x, peak, mean, capsigma, r2, r4, 0, 0)


@nb.njit()
def polyg4const(x, peak, mean, capsigma, r2, r4, const):
    return polyg6const(x, peak, mean, capsigma, r2, r4, 0, const)


@nb.njit()
def polyg2(x, peak, mean, capsigma, r2):
    return polyg6const(x, peak, mean, capsigma, r2, 0, 0, 0)


@nb.njit()
def polyg2const(x, peak, mean, capsigma, r2, const):
    return polyg6const(x, peak, mean, capsigma, r2, 0, 0, const)


@nb.njit()
def twomudg(x, A, frac, sigma1, sigma2, mean1, mean2):
    return A *(frac * sg(x, 1, mean1, sigma1) + (1 - frac) * sg(x, 1, mean1 + mean2, sigma2))


# Each function needs a mapping from string given as a parameter
fit_functions = {
    'sg':          sg,
    'sgConst':     sgconst,
    'dg':          dg,
    'dgConst':     dgconst,
    'polyG6':      polyg6,
    'polyG6Const': polyg6const,
    'polyG4':      polyg4,
    'polyG4Const': polyg4const,
    'polyG2':      polyg2,
    'polyG2Const': polyg2const,
    'superG':      superg,
    'superGConst': supergconst,
    'twoMuDg':     twomudg
}

