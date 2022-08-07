import yaml
import matplotlib.pyplot as plt
import numpy as np

import fits


CONFIG_FILENAME = 'config.yml'
with open(CONFIG_FILENAME, 'r') as cfg:
    CONFIG = yaml.safe_load(cfg)

x = np.linspace(-1,1,100)

plt.figure()

for name, handle in fits.fit_functions.items():
    param = CONFIG['fitting']['parameter_initial_values'][name]
    if param['peak'] == 'auto':
        param['peak'] = 1
    plt.plot(handle(x, **param), label=name)

plt.legend()
plt.show()
