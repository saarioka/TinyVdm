import numpy as np
import mplhep as hep
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from jacobi import propagate

import fits

matplotlib.use('Agg')

plt.style.use(hep.style.CMS)


def as_si(x, ndp=2):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    if 'e' in s:
        m, e = s.split('e')
        return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
    else:
        return str(x)


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

    def create_page(self, x, y, yerr, fit, info, covariance=None, *argv):
        figure_items = []  # save handles here to be able to delete them without affecting template
        figure_items.append(self.ax1.errorbar(x, y, yerr, fmt='k.', linewidth=2, elinewidth=2, capsize=5, capthick=2)) # Plot the data points

        x_dense = np.linspace(np.min(x), np.max(x))

        if covariance is not None:
            yy, ycov = propagate(lambda p: fits.fit_functions[fit](x_dense, *p), argv, covariance)
            yerr_prop = np.diag(ycov) ** 0.5
            figure_items.append(self.ax1.fill_between(x_dense, yy - yerr_prop, yy + yerr_prop, facecolor="k", alpha=0.3))

        figure_items.append(self.ax1.plot(x_dense, fits.fit_functions[fit](x_dense, *argv), 'k')) # Plot the fit result

        figure_items.append(self.ax1.text(0.97, 0.97, info, transform=self.ax1.transAxes, fontsize=14, fontweight='bold',
                            verticalalignment='top', horizontalalignment='right'))

        residuals = (y.to_numpy() - fits.fit_functions[fit](x, *argv).to_numpy()) / yerr.to_numpy()
        figure_items.append(self.ax2.errorbar(x, residuals, 1, fmt='k.', linewidth=2, elinewidth=2, capsize=5, capthick=2))

        # plot without changing xlim
        lim = list(plt.xlim())
        figure_items.append(self.ax2.plot(lim, [0, 0], 'k:'))
        plt.xlim(lim)

        self.ax1.relim()
        self.ax1.autoscale()
        self.ax2.relim()

        self.pdf.savefig()

        for item in figure_items:  # Only delete lines and fit results, leave general things
            if isinstance(item, list):
                item[0].remove()
            else:
                item.remove()

    def close_pdf(self):
        self.pdf.close()
        plt.close(self.fig)

