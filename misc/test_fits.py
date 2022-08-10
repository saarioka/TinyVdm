import pylab
import yaml
import matplotlib.pyplot as plt
import numpy as np

import fits

CONFIG_FILENAME = 'config.yml'
with open(CONFIG_FILENAME, 'r') as cfg:
    CONFIG = yaml.safe_load(cfg)

def polyg6const(x, peak, mean, capsigma, r2, r4, r6, const):
    x0 = x-mean
    sigma = capsigma / (1 + r2 + 3*r4 + 15*r6)
    return const + peak * (1 + r2*(x0/sigma)**2 + r4*(x0/sigma)**4 + r6*(x0/sigma)**6) * np.exp(-(x0/sigma)**2/2)

def polyg6(x, peak, mean, capsigma, r2, r4, r6):
    print(peak, mean, capsigma, r2, r4, r6)
    return polyg6const(x, peak, mean, capsigma, r2, r4, r6, 0)


def f(S, Gmax, Km):
    s1 = Gmax*S   # G_max
    e1 = S + Km  # K_m
    return np.divide(s1, e1)


def update(val):
    l.set_ydata(f(S, sGmax.val, sKm.val))

fit = 'polyG6'
#handle = fits.fit_functions[fit]
handle = polyg6

init = CONFIG['fitting']['parameter_initial_values'][fit]
if init['peak'] == 'auto':
    init['peak'] = 1
if init['capsigma'] == 'auto':
    init['capsigma'] = 1

print(init)

x = np.linspace(-1,1,100)

#plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.subplots_adjust(left=0.15, bottom=0.25)
l, = plt.plot(x, handle(x, **init))

plt.grid(False)
plt.title('Playing with sliders')
plt.xlabel('time')
plt.ylabel('concentration')

axcolor = 'lightgoldenrodyellow'
axGmax = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
axKm = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)

plt.show()

sGmax = pylab.Slider(axGmax, 'Gmax', 0.1, 3.0, valinit=1)
sKm = pylab.Slider(axKm, 'Km', 0.01, 1.0, valinit=1)

sGmax.on_changed(update)
sKm.on_changed(update)

plt.show()


peak = 1
mean = 0
capsigma = 1
r2 = 0
r4 = 0
r6 = 0

plt.figure(figsize=(12,8))

plt.plot(x, fits.polyg6(x, peak, mean, capsigma, r2, r4, r6))

plt.legend()
plt.show()
