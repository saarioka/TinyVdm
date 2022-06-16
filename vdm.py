from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import tables
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.special import gamma
from iminuit import Minuit
from iminuit.cost import LeastSquares

matplotlib.use('Agg')

plt.style.use(hep.style.CMS)

# Define functions that may be fit to the rates
def sg(x, peak, mean, cap_sigma):
    return peak*np.exp(-(x-mean)**2/(2*cap_sigma**2))

def sg_const(x, peak, mean, cap_sigma, constant):
    return sg(x, peak, mean, cap_sigma) + constant

def super_gauss(x, peak, mean, cap_sigma, p):
    beta = p*2.0
    alpha = np.sqrt(2*np.pi)*beta / (2*gamma(1.0/beta))*cap_sigma
    return peak*np.exp(-(np.abs(x-mean)/alpha)**beta)

def poly_gauss(x, peak, mean, cap_sigma, r2, r4):
    x0 = x - peak
    sigma = cap_sigma / (1 + r2 + 3*r4)
    return peak * (1 + r2*(x0/sigma)**2 + r4*(x0/sigma)**4 ) * np.exp(-(x0/sigma)**2/2)

def dg(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio):
    return sg(x, peak*peak_ratio, mean, cap_sigma*cap_sigma_ratio) + sg(x, peak*(1-peak_ratio), mean, cap_sigma*(1-cap_sigma_ratio))

def dg_const(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio, constant):
    return dg(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio) + constant

# Each function needs a mapping from string given as a parameter, and also a set of initial conditions
FIT_FUNCTIONS = {
    'sg':          {'handle': sg,          'initial_values': {'peak': 1e-4, 'mean': 0, 'cap_sigma': 0.3}},
    'sg_const':    {'handle': sg_const,    'initial_values': {'peak': 1e-4, 'mean': 0, 'cap_sigma': 0.3, 'constant': 0}},
    'super_gauss': {'handle': super_gauss, 'initial_values': {'peak': 1e-4, 'mean': 0, 'cap_sigma': 0.3, 'p': 1}},
    'poly_gauss':  {'handle': poly_gauss,  'initial_values': {'peak': 1e-4, 'mean': 0, 'cap_sigma': 0.3, 'r2': 0.1, 'r4': 0.1}},
    'dg':          {'handle': dg,          'initial_values': {'peak': 2e-4, 'mean': 0, 'cap_sigma': 0.4, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5}},
    'dg_const':    {'handle': dg_const,    'initial_values': {'peak': 2e-4, 'mean': 0, 'cap_sigma': 0.4, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5, 'constant': 0}}
}

def get_fbct_to_dcct_correction_factors(f, period_of_scanpoint, filled):
    fbct_b1 = np.array([b['bxintensity1'][filled] for b in f.root['beam'].where(period_of_scanpoint)])
    fbct_b2 = np.array([b['bxintensity2'][filled] for b in f.root['beam'].where(period_of_scanpoint)])
    dcct = np.array([[b['intensity1'], b['intensity2']] for b in f.root['beam'].where(period_of_scanpoint)]) # Normalised beam current
    return np.array([fbct_b1.sum(), fbct_b2.sum()]) / dcct.sum(axis=0)

def bkg_from_noncolliding(f, period_of_scanpoint, luminometer): # WIP
    """Not working"""
    abort_gap_mask = [*range(3444, 3564)]
    filled_noncolliding = np.nonzero(np.logical_xor(f.root.beam[0]['bxconfig1'], f.root.beam[0]['bxconfig2']))[0]
    rate_nc = np.array([r['bxraw'][filled_noncolliding] for r in f.root[luminometer].where(period_of_scanpoint)])
    rate_ag = np.array([r['bxraw'][abort_gap_mask] for r in f.root[luminometer].where(period_of_scanpoint)])
    bkg = 2*rate_nc.mean() - rate_ag.mean()
    return bkg

def main(args):
    for filename in args.files:
        outpath = f'output/{Path(filename).stem}' # Save output to this folder
        Path(outpath).mkdir(parents=True, exist_ok=True) # Create output folder if not existing already
        with tables.open_file(filename, 'r') as f:
            collidable = np.nonzero(f.root.beam[0]['collidable'])[0] # indices of colliding bunches (0-indexed)
            filled = np.nonzero(np.logical_or(f.root.beam[0]['bxconfig1'], f.root.beam[0]['bxconfig2']))[0]

            # Associate timestamps to scan plane - scan point -pairs
            scan = pd.DataFrame()
            scan['timestampsec'] = [r['timestampsec'] for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
            scan['sep'] = [r['sep'] for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
            scan['nominal_sep_plane'] = [r['nominal_sep_plane'].decode('utf-8') for r in f.root.vdmscan.where('stat == "ACQUIRING"')] # Decode is needed for values of type string
            scan = scan.groupby(['nominal_sep_plane', 'sep']).agg(min_time=('timestampsec', np.min), max_time=('timestampsec', np.max)) # Get min and max for each plane - sep pair
            scan.reset_index(inplace=True)
            scan_info = pd.DataFrame([list(f.root.vdmscan[0])], columns=f.root.vdmscan.colnames) # Get first row of table "vdmscan" to save scan conditions that are constant thorugh the scan
            scan_info['ip'] = scan_info['ip'].apply(lambda ip: [i for i,b in enumerate(bin(ip)[::-1]) if b == '1']) # Binary to dec to list all scanning IPs
            scan_info['energy'] = f.root.beam[0]['egev']
            scan_info[['fillnum', 'runnum', 'timestampsec', 'energy', 'ip', 'bstar5', 'xingHmurad']].to_csv(f'{outpath}/scan.csv', index=False) # Save this set of conditions to file

            rate_and_beam = pd.DataFrame()
            for p, plane in enumerate(scan.nominal_sep_plane.unique()):
                for b, sep in enumerate(scan.sep.unique()):
                    period_of_scanpoint = f'(timestampsec > {scan.min_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()}) & (timestampsec <= {scan.max_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()})'

                    r = np.array([r['bxraw'][collidable] for r in f.root[args.luminometer].where(period_of_scanpoint)]) # Rates of colliding bcids for this scan point
                    beam = np.array([b['bxintensity1'][collidable]*b['bxintensity2'][collidable]/1e22 for b in f.root['beam'].where(period_of_scanpoint)]) # Normalised beam current

                    new_data = pd.DataFrame(np.array([r.mean(axis=0), stats.sem(r, axis=0), beam.mean(axis=0)]).T, columns=['rate', 'rate_err', 'beam']) # Mean over lumi sections
                    new_data['fbct_dcct_fraction_b1'], new_data['fbct_dcct_fraction_b2'] = get_fbct_to_dcct_correction_factors(f, period_of_scanpoint, filled)

                    new_data.insert(0, 'bcid', collidable+1) # Move to 1-indexed values of BCID
                    new_data.insert(0, 'sep', sep)
                    new_data.insert(0, 'plane', plane)

                    rate_and_beam = new_data if p == 0 and b == 0 else pd.concat([rate_and_beam, new_data])


        rate_and_beam['rate_normalised'] = rate_and_beam.rate / rate_and_beam.beam
        rate_and_beam['rate_normalised_err'] = rate_and_beam.rate_err / rate_and_beam.beam

        if not args.no_fbct_dcct:
            calib = rate_and_beam.groupby('plane')[['fbct_dcct_fraction_b1', 'fbct_dcct_fraction_b2']].transform('mean').prod(axis=1) # Mean over LS, multiply B1 * B2
            rate_and_beam['beam_calibrated'] = rate_and_beam.beam / calib
            rate_and_beam['rate_normalised'] *= calib
            rate_and_beam['rate_normalised_err'] *= calib

        rate_and_beam['rate_normalised_err'].replace(0, rate_and_beam['rate_normalised_err'].max(axis=0), inplace=True) # Add sensible error in case of 0 rate (max of error)

        rate_and_beam.to_csv(f'{outpath}/rate_and_beam.csv', index=False)

        if args.pdf: # initialize template for plots
            pdf = PdfPages(f'{outpath}/fit_{args.luminometer}.pdf')

            fig = plt.figure()
            ax1 = fig.add_axes((.12,.3,.83,.65)) # Upper part: fit and data points
            hep.cms.label(llabel="Preliminary", rlabel=fr"Fill {scan_info.fillnum[0]}, $\sqrt{{s}}={scan_info['energy'][0]*2/1000:.1f}$ TeV", loc=1)
            ax1.set_ylabel('$R/(N_1 N_2)$ [arb.]')
            ax1.set_xticklabels([])
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True, useOffset=False)
            ax1.minorticks_off()

            ax2 = fig.add_axes((.12,.1,.83,.2)) # Lower part: residuals
            ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax2.set_ylabel('Residual [$\sigma$]',fontsize=20)
            ax2.set_xlabel('$\Delta$ [mm]')
            ax2.minorticks_off()

        for p, plane in enumerate(rate_and_beam.plane.unique()): # For each plane
            for b, bcid in enumerate(rate_and_beam.bcid.unique()): # For each BCID
                data_x = scan[scan.nominal_sep_plane == plane]['sep'] # x-data: separations
                data_y = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid)]['rate_normalised'] # y-data: normalised rates
                data_y_err = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid)]['rate_normalised_err']

                least_squares = LeastSquares(data_x, data_y, data_y_err, FIT_FUNCTIONS[args.fit]['handle']) # Initialise minimiser with data and fit function of choice
                m = Minuit(least_squares, **FIT_FUNCTIONS[args.fit]['initial_values']) # Give the initial values defined in "FIT_FUNCTIONS"
                m.limits['cap_sigma'] = [0, None] # Most preferably positive beam widths
                m.migrad()  # Finds minimum of least_squares function
                m.hesse()   # Accurately computes uncertainties

                new = pd.DataFrame([m.values], columns=m.parameters) # Store values and errors to dataframe
                new = pd.concat([new, pd.DataFrame([m.errors], columns=m.parameters).add_suffix('_err')], axis=1) # Add suffix "_err" to errors
                new['valid'] =  m.valid
                new['accurate'] = m.accurate
                new.insert(0, 'bcid', bcid)
                new.insert(0, 'plane', plane)

                fit_results = new if b == 0 and p == 0 else pd.concat([fit_results, new], ignore_index=True)

                if args.pdf:
                    figure_items = []
                    figure_items.append(ax1.errorbar(data_x, data_y, data_y_err, fmt='ko')) # Plot the data points
                    x_dense = np.linspace(np.min(data_x), np.max(data_x))
                    figure_items.append(ax1.plot(x_dense, FIT_FUNCTIONS[args.fit]['handle'](x_dense, *m.values), 'k')) # Plot the fit result

                    fit_info = [f'{plane}, BCID {bcid}', f'$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}']
                    for param, v, e in zip(m.parameters, m.values, m.errors):
                        fit_info.append(f'{param} = ${v:.3e} \\pm {e:.3e}$')

                    fit_info = [info.replace('cap_sigma', '$\Sigma$') for info in fit_info]

                    figure_items.append(ax1.text(0.95, 0.95, '\n'.join(fit_info), transform=ax1.transAxes, fontsize=14, fontweight='bold',
                        verticalalignment='top', horizontalalignment='right'))

                    residuals = (data_y.to_numpy() - FIT_FUNCTIONS[args.fit]['handle'](data_x, *m.values).to_numpy()) / data_y_err.to_numpy()
                    figure_items.append(ax2.scatter(data_x, residuals, c='k'))
                    lim = list(plt.xlim()); figure_items.append(ax2.plot(lim, [0, 0], 'k:')); plt.xlim(lim) # plot without changing xlim

                    pdf.savefig()

                    for item in figure_items: # Only delete lines and fit results, leave general things
                        if isinstance(item, list):
                            item[0].remove()
                        else:
                            item.remove()
        if args.pdf:
            pdf.close()

        fit_results.cap_sigma *= 1e3 # to µm
        fit_results.cap_sigma_err *= 1e3 # to µm
        fit_results.to_csv(f'{outpath}/{args.luminometer}_fit_results.csv', index=False)

        val = fit_results.pivot(index='bcid', columns=['plane'], values=['cap_sigma', 'peak', 'cap_sigma_err', 'peak_err'])

        sigvis = np.pi * val.cap_sigma.prod(axis=1) * val.peak.sum(axis=1)

        sigvis_err = (val.cap_sigma_err**2 / val.cap_sigma**2).sum(axis=1) + (val.peak_err**2).sum(axis=1) / (val.peak).sum(axis=1)**2
        sigvis_err = np.sqrt(sigvis_err) * sigvis

        lumi = pd.concat([sigvis, sigvis_err], axis=1)
        lumi.columns = ['sigvis', 'sigvis_err']
        lumi.to_csv(f'{outpath}/lumi.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--luminometer', type=str, help='Luminometer name', required=True)
    parser.add_argument('-nofd', '--no_fbct_dcct', help='Do NOT calibrate beam current', action='store_true')
    parser.add_argument('-bkg', '--background_correction', help='Apply bckground correction', action='store_true')
    parser.add_argument('-pdf', '--pdf', help='Create fit PDFs', action='store_true')
    parser.add_argument('-fit', '--fit', type=str, help='Fit function', choices=FIT_FUNCTIONS.keys(), default='sg')
    parser.add_argument('files', nargs='*')

    main(parser.parse_args())

