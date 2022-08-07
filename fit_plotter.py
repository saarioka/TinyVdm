import yaml
import numpy as np
import mplhep as hep
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from jacobi import propagate

import fits

matplotlib.use('Agg')

plt.style.use(hep.style.CMS)

COLOR1 = '#3c3b6e'
COLOR2 = '#b22234'

CONFIG_FILENAME = 'config.yml'
with open(CONFIG_FILENAME, 'r') as cfg:
    CONFIG = yaml.safe_load(cfg)


def as_si(x, ndp=2):
    """Convert number to latex scientific format if necessary"""
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    if 'e' in s:
        m, e = s.split('e')
        return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
    return str(x)


class plotter():
    def __init__(self, filename, fill=None, energy=None):
        self.pdf = PdfPages(filename)

        self.fig = plt.figure()

        # Upper part: fit and data points
        self.ax1 = self.fig.add_axes((.12,.3,.81,.65))
        hep.cms.label(llabel="Preliminary", rlabel=fr"Fill {fill}, $\sqrt{{s}}={energy:.1f}$ TeV", loc=1, fontsize=20)
        self.ax1.set_ylabel('$R/(N_1 N_2)$ [arb.]')
        self.ax1.set_xticklabels([])
        self.ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True, useOffset=False)
        self.ax1.minorticks_off()

        self.ax1_log = self.ax1.twinx()
        self.ax1_log.set_yscale("log")
        self.ax1_log.set_xticklabels([])
        self.ax1_log.minorticks_off()

        # Lower part: residuals
        self.ax2 = self.fig.add_axes((.12,.1,.81,.2))
        self.ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
        self.ax2.set_ylabel(r'Residual [$\sigma$]',fontsize=20)
        self.ax2.set_xlabel(r'$\Delta$ [mm]')
        self.ax2.minorticks_off()

        self.handles = {}

    def create_page(self, x, y, yerr, fit, info, covariance=None, *argv):
        figure_items = []  # save handles here to be able to delete them without affecting template

        x_dense = np.linspace(np.min(x), np.max(x))

        residuals = (y.to_numpy() - fits.fit_functions[fit](x, *argv).to_numpy()) / yerr.to_numpy()

        # https://stackoverflow.com/questions/15887820/animation-by-using-matplotlib-errorbar
        # https://stackoverflow.com/questions/25210723/matplotlib-set-data-for-errorbar-plot
        for key, val in self.handles.items():
            if key == 'errorbars':
                for _, (line, (bottoms, tops), verts) in val.items():
                    line.set_ydata(y)
                    bottoms.set_ydata(y - yerr)
                    tops.set_ydata(y + yerr)
                    barsy, = verts
                    barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y + yerr, y - yerr)])
            if key == 'lines':
                for _, line in val.items():
                    line[0].set_ydata(fits.fit_functions[fit](x_dense, *argv))
            if key == 'residuals':
                line, (bottoms, tops), verts = val
                line.set_ydata(residuals)
                bottoms.set_ydata(residuals - 1)
                tops.set_ydata(residuals + 1)
                barsy, = verts
                barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, residuals + 1, residuals - 1)])


        if not self.handles:
            self.handles = {'errorbars': {}, 'lines': {}}

            # Plot the data points
            self.handles['errorbars']['datapoints'] = self.ax1.errorbar(x, y, yerr, fmt='.', color=COLOR1, linewidth=2, elinewidth=2, capsize=5, capthick=2)
            self.handles['errorbars']['datapoints_log'] = self.ax1_log.errorbar(x, y, yerr, fmt='.', color=COLOR2, linewidth=2, elinewidth=2, capsize=5, capthick=2)

            # Plot the fit result
            self.handles['lines']['fit'] = self.ax1.plot(x_dense, fits.fit_functions[fit](x_dense, *argv), color=COLOR1)
            self.handles['lines']['fit_log'] = self.ax1_log.plot(x_dense, fits.fit_functions[fit](x_dense, *argv), color=COLOR2)

            # Plot the residuals
            self.handles['residuals'] = self.ax2.errorbar(x, residuals, 1, fmt='k.', linewidth=2, elinewidth=2, capsize=5, capthick=2)


        if CONFIG['plotting']['display_fit_uncertainty'] and covariance is not None:
            yy, ycov = propagate(lambda p: fits.fit_functions[fit](x_dense, *p), argv, covariance)
            yerr_prop = np.diag(ycov) ** 0.5
            figure_items.append(self.ax1.fill_between(x_dense, yy - yerr_prop, yy + yerr_prop, facecolor=COLOR1, alpha=0.3))
            figure_items.append(self.ax1_log.fill_between(x_dense, yy - yerr_prop, yy + yerr_prop, facecolor=COLOR2, alpha=0.3))

        figure_items.append(self.ax1.text(0.65, 0.03, info, transform=self.ax1.transAxes, fontsize=14, fontweight='bold',
                            verticalalignment='bottom', horizontalalignment='right'))

        # plot without changing xlim
        lim = list(plt.xlim())
        figure_items.append(self.ax2.plot(lim, [0, 0], 'k', linewidth=0.5))
        plt.xlim(lim)

        self.ax1.relim()
        self.ax1.autoscale()
        self.ax1_log.relim()
        self.ax1_log.autoscale()
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

