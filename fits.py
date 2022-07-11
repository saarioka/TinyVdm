import numpy as np
from scipy.special import gamma


def sgconst(x, peak, mean, capsigma, const):
    return const + peak*np.exp(-(x-mean)**2/(2*capsigma**2))


def sg(x, peak, mean, capsigma):
    return sgconst(x, peak, mean, capsigma, 0)


def dgconst(x, peak, mean, capsigma, peak_ratio, capsigma_ratio, const):
    #sigma = capsigma * capsigma_ratio / (peak_ratio*capsigma_ratio + 1 - peak_ratio)
    #peak = capsigma / (peak_ratio*capsigma_ratio + 1 - peak_ratio)
    c = capsigma / (peak_ratio*capsigma_ratio + 1 - peak_ratio)
    return const + sg(x, peak_ratio*peak, mean, capsigma_ratio*c) + sg(x, (1-peak_ratio)*peak, mean, c)


def dg(x, peak, mean, capsigma, peak_ratio, capsigma_ratio):
    return dgconst(x, peak, mean, capsigma, peak_ratio, capsigma_ratio, 0)


def polyg2const(x, peak, mean, capsigma, r2, const):
    return polyg6const(x, peak, mean, capsigma, r2, 0, 0, const)


def supergconst(x, peak, mean, capsigma, p, const):
    beta = p*2.0
    alpha = np.sqrt(2*np.pi)*beta / (2*gamma(1.0/beta))*capsigma
    return const + peak*np.exp(-(np.abs(x-mean)/alpha)**beta)


def superg(x, peak, mean, capsigma, p):
    return supergconst(x, peak, mean, capsigma, p, 0)


def polyg6const(x, peak, mean, capsigma, r2, r4, r6, const):
    x0 = x-mean
    sigma = capsigma / (1 + r2 + 3*r4 + 15*r6)
    return const + peak * (1 + r2*(x0/sigma)**2 + r4*(x0/sigma)**4 + r6*(x0/sigma)**6) * np.exp(-(x0/sigma)**2/2)


def polyg6(x, peak, mean, capsigma, r2, r4, r6):
    return polyg6const(x, peak, mean, capsigma, r2, r4, r6, 0)


def polyg4(x, peak, mean, capsigma, r2, r4):
    return polyg6const(x, peak, mean, capsigma, r2, r4, 0, 0)


def polyg4const(x, peak, mean, capsigma, r2, r4, const):
    return polyg6const(x, peak, mean, capsigma, r2, r4, 0, const)


def polyg2(x, peak, mean, capsigma, r2):
    return polyg6const(x, peak, mean, capsigma, r2, 0, 0, 0)


# Each function needs a mapping from string given as a parameter
fit_functions = {
    'sg':          sg,
    'sgConst':     sgconst,
    'dg':          dg,
    'dgConst':     dgconst,
    'polyG6':      polyg6,
    'polyG6Const': polyg6const,
    'polyG4':      polyg4,
    'polyG4onst':  polyg4const,
    'polyG2':      polyg2,
    'polyG2Const': polyg2const,
    'superG':      superg,
    'superGConst': supergconst
}

