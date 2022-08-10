import pylab
import yaml
import matplotlib.pyplot as plt
import numpy as np

import fits

CONFIG_FILENAME = 'config.yml'
with open(CONFIG_FILENAME, 'r') as cfg:
    CONFIG = yaml.safe_load(cfg)

fit = 'polyG6'
handle = fits.fit_functions[fit]

x = np.linspace(-1,1,100)

peak = 6e-5
mean = 0
capsigma = 3.7e-1
r2 = -2e-1
r4 = -6.1e-2
r6 = 1.25e-2

plt.figure(figsize=(12,8))

plt.plot(x, handle(x, peak, mean, capsigma, r2, r4, r6))
plt.gca().set_yscale('log')

plt.legend()
plt.show()
