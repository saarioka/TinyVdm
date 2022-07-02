import numpy as np
import mplhep as hep
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fits

matplotlib.use('Agg')

plt.style.use(hep.style.CMS)

class plotter():
    def __init__(self, filename, fill=None, energy=None):
        self.pdf = PdfPages(filename)

        self.fig = plt.figure()
        self.ax1 = self.fig.add_axes((.12,.3,.83,.65))  # Upper part: fit and data points

        hep.cms.label(llabel="Preliminary", rlabel=fr"Fill {fill}, $\sqrt{{s}}={energy:.1f}$ TeV", loc=1)

        self.ax1.set_ylabel('$R/(N_1 N_2)$ [arb.]')
        self.ax1.set_xticklabels([])
        self.ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True, useOffset=False)
        self.ax1.minorticks_off()

        self.ax2 = self.fig.add_axes((.12,.1,.83,.2))  # Lower part: residuals
        self.ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
        self.ax2.set_ylabel(r'Residual [$\sigma$]',fontsize=20)
        self.ax2.set_xlabel(r'$\Delta$ [mm]')
        self.ax2.minorticks_off()

    def __del__(self):
        self.pdf.close()
        plt.close(self.fig)

    def create_page(self, x, y, yerr, fit, info, *argv):
        figure_items = []  # save handles here to be able to delete them without affecting template
        figure_items.append(self.ax1.errorbar(x, y, yerr, fmt='k.', capsize=5)) # Plot the data points

        x_dense = np.linspace(np.min(x), np.max(x))
        figure_items.append(self.ax1.plot(x_dense, fits.fit_functions[fit]['handle'](x_dense, *argv), 'k')) # Plot the fit result

        figure_items.append(self.ax1.text(0.95, 0.95, info, transform=self.ax1.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', horizontalalignment='right'))

        residuals = (y.to_numpy() - fits.fit_functions[fit]['handle'](x, *argv).to_numpy()) / yerr.to_numpy()
        figure_items.append(self.ax2.errorbar(x, residuals, 1, fmt='k.', capsize=5))

        # plot without changing xlim
        lim = list(plt.xlim())
        figure_items.append(self.ax2.plot(lim, [0, 0], 'k:'))
        plt.xlim(lim)

        self.pdf.savefig()

        for item in figure_items:  # Only delete lines and fit results, leave general things
            if isinstance(item, list):
                item[0].remove()
            else:
                item.remove()

