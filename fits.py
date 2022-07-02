import numpy as np
from scipy.special import gamma


def sg(x, peak, mean, capsigma):
    return peak*np.exp(-(x-mean)**2/(2*capsigma**2))


def sgconst(x, peak, mean, capsigma, constant):
    return sg(x, peak, mean, capsigma) + constant


def dg(x, peak, mean, capsigma, peak_ratio, capsigma_ratio):
    return sg(x, peak*peak_ratio, mean, capsigma*capsigma_ratio) + sg(x, peak*(1-peak_ratio), mean, capsigma*(1-capsigma_ratio))


def dgconst(x, peak, mean, capsigma, peak_ratio, capsigma_ratio, constant):
    return dg(x, peak, mean, capsigma, peak_ratio, capsigma_ratio) + constant


def polyg6(x, peak, mean, capsigma, r2, r4, r6):
    x0 = x-mean
    sigma = capsigma / (1 + r2 + 3*r4 + 15*r6)
    return peak*(1 + r2*(x0/sigma)**2 + r4*(x0/sigma)**4 + r6*(x0/sigma)**6)*np.exp(-(x0/sigma)**2/2)


def polyg6const(x, peak, mean, capsigma, r2, r4, r6, const):
    return polyg6(x, peak, mean, capsigma, r2, r4, r6) + const


def polyg4(x, peak, mean, capsigma, r2, r4):
    return polyg6(x, peak, mean, capsigma, r2, r4, 0)


def polyg4const(x, peak, mean, capsigma, r2, r4, const):
    return polyg6const(x, peak, mean, capsigma, r2, r4, 0, const)


def polyg2(x, peak, mean, capsigma, r2):
    return polyg4(x, peak, mean, capsigma, r2, 0)


def polyg2const(x, peak, mean, capsigma, r2, const):
    return polyg4const(x, peak, mean, capsigma, r2, 0, const)


def superg(x, peak, mean, capsigma, p):
    beta = p*2.0
    alpha = np.sqrt(2*np.pi)*beta / (2*gamma(1.0/beta))*capsigma
    return peak*np.exp(-(np.abs(x-mean)/alpha)**beta)


def supergconst(x, peak, mean, capsigma, p, const):
    return superg(x, peak, mean, capsigma, p) + const


# Each function needs a mapping from string given as a parameter, and also a set of initial conditions
fit_functions = {
    'sg':           {'handle': sg,          'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3}},
    'sgConst':      {'handle': sgconst,     'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'constant': 0}},
    'dg':           {'handle': dg,          'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.2, 'peak_ratio': 0.5, 'capsigma_ratio': 2}},
    'dgConst':      {'handle': dgconst,     'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.2, 'peak_ratio': 0.5, 'capsigma_ratio': 2, 'const': 0}},
    'polyG6':       {'handle': polyg6,      'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'r2': 0, 'r4': 0, 'r6': 0}},
    'polyG6Const':  {'handle': polyg6const, 'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'r2': 0, 'r4': 0, 'r6': 0, 'const': 0}},
    'polyG4':       {'handle': polyg4,      'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'r2': 0, 'r4': 0}},
    'polyG4onst':   {'handle': polyg4const, 'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'r2': 0, 'r4': 0, 'const': 0}},
    'polyG2':       {'handle': polyg2,      'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'r2': 0}},
    'polyG2Const':  {'handle': polyg2const, 'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'r2': 0, 'const': 0}},
    'superG':       {'handle': superg,      'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'p': 1}},
    'superGConst':  {'handle': supergconst, 'initial_values': {'peak': 'auto', 'mean': 0, 'capsigma': 0.3, 'p': 1, 'const': 0}},
}

parameter_limits = {
    'peak': [0, None],
    'capsigma': [1e-3, 2],
    'mean': [-1e-2, 1e-2],
    'peak_ratio': [0, 1],
    'capsigma_ratio': [2, None]
}

