from pathlib import Path
import argparse

import shutil
import os
import tables
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplhep as hep
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


def sgconst(x, peak, mean, cap_sigma, constant):
    return sg(x, peak, mean, cap_sigma) + constant


def dg(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio):
    return sg(x, peak*peak_ratio, mean, cap_sigma*cap_sigma_ratio) + sg(x, peak*(1-peak_ratio), mean, cap_sigma*(1-cap_sigma_ratio))


def dgconst(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio, constant):
    return dg(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio) + constant


def polyg6(x, peak, mean, cap_sigma, r2, r4, r6):
    x0 = x-mean
    sigma = cap_sigma / (1 + r2 + 3*r4 + 15*r6)
    return peak*(1 + r2*(x0/sigma)**2 + r4*(x0/sigma)**4 + r6*(x0/sigma)**6)*np.exp(-(x0/sigma)**2/2)


def polyg6const(x, peak, mean, cap_sigma, r2, r4, r6, const):
    return polyg6(x, peak, mean, cap_sigma, r2, r4, r6) + const


def polyg4(x, peak, mean, cap_sigma, r2, r4):
    return polyg6(x, peak, mean, cap_sigma, r2, r4, 0)


def polyg4const(x, peak, mean, cap_sigma, r2, r4, const):
    return polyg6const(x, peak, mean, cap_sigma, r2, r4, 0, const)


def polyg2(x, peak, mean, cap_sigma, r2):
    return polyg4(x, peak, mean, cap_sigma, r2, 0)


def polyg2const(x, peak, mean, cap_sigma, r2, const):
    return polyg4const(x, peak, mean, cap_sigma, r2, 0, const)


def superg(x, peak, mean, cap_sigma, p):
    beta = p*2.0
    alpha = np.sqrt(2*np.pi)*beta / (2*gamma(1.0/beta))*cap_sigma
    return peak*np.exp(-(np.abs(x-mean)/alpha)**beta)


def supergconst(x, peak, mean, cap_sigma, p, const):
    return superg(x, peak, mean, cap_sigma, p) + const


# Each function needs a mapping from string given as a parameter, and also a set of initial conditions
FIT_FUNCTIONS = {
    'sg':           {'handle': sg,          'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3}},
    'sgConst':      {'handle': sgconst,     'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'constant': 0}},
    'dg':           {'handle': dg,          'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5}},
    'dgConst':      {'handle': dgconst,     'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5, 'const': 0}},
    'polyG6':       {'handle': polyg6,      'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'r2': 0, 'r4': 0, 'r6': 0}},
    'polyG6Const':  {'handle': polyg6const, 'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'r2': 0, 'r4': 0, 'r6': 0, 'const': 0}},
    'polyG4':       {'handle': polyg4,      'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'r2': 0, 'r4': 0}},
    'polyG4onst':   {'handle': polyg4const, 'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'r2': 0, 'r4': 0, 'const': 0}},
    'polyG2':       {'handle': polyg2,      'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'r2': 0}},
    'polyG2Const':  {'handle': polyg2const, 'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'r2': 0, 'const': 0}},
    'superG':       {'handle': superg,      'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'p': 1}},
    'superGConst':  {'handle': supergconst, 'initial_values': {'peak': 1e-2, 'mean': 0, 'cap_sigma': 0.3, 'p': 1, 'const': 0}},
}

PARAMETER_LIMITS = {'peak': [0, None], 'cap_sigma': [0, 2], 'mean': [-1e-1, 1e-1]}


def get_fbct_to_dcct_correction_factors(f, period_of_scanpoint, filled):
    fbct_b1 = np.array([b['bxintensity1'][filled] for b in f.root['scan5_beam'].where(period_of_scanpoint)])
    fbct_b2 = np.array([b['bxintensity2'][filled] for b in f.root['scan5_beam'].where(period_of_scanpoint)])
    dcct = np.array([[b['intensity1'], b['intensity2']] for b in f.root['scan5_beam'].where(period_of_scanpoint)]) # Normalised beam current
    return np.array([fbct_b1.sum(), fbct_b2.sum()]) / dcct.sum(axis=0)


def bkg_from_noncolliding(f, period_of_scanpoint, luminometer): # WIP
    """Not working"""
    abort_gap_mask = [*range(3444, 3564)]
    filled_noncolliding = np.nonzero(np.logical_xor(f.root.scan5_beam[0]['bxconfig1'], f.root.scan5_beam[0]['bxconfig2']))[0]
    rate_nc = np.array([r['bxraw'][filled_noncolliding] for r in f.root[luminometer].where(period_of_scanpoint)])
    rate_ag = np.array([r['bxraw'][abort_gap_mask] for r in f.root[luminometer].where(period_of_scanpoint)])
    bkg = 2*rate_nc.mean() - rate_ag.mean()
    return bkg


def get_scan_info(filename):
    with tables.open_file(filename, 'r') as f:
        # Associate timestamps to scan plane - scan point -pairs

        quality_condition = '(stat == "ACQUIRING") & (nominal_sep_plane != "NONE")'
        scan = pd.DataFrame()
        scan['timestampsec'] = [r['timestampsec'] for r in f.root.vdmscan.where(quality_condition)]
        scan['sep'] = [r['sep'] for r in f.root.vdmscan.where(quality_condition)]

        # Decode is needed for values of type string
        scan['nominal_sep_plane'] = [r['nominal_sep_plane'].decode('utf-8') for r in f.root.vdmscan.where(quality_condition)]

        # Get min and max for each plane - sep pair
        scan = scan.groupby(['nominal_sep_plane', 'sep']).agg(min_time=('timestampsec', np.min), max_time=('timestampsec', np.max))

    return scan.reset_index()


def get_basic_info(filename):
    with tables.open_file(filename, 'r') as f:
        # Get first row of table "vdmscan" to save scan conditions that are constant thorugh the scan
        scan_info = pd.DataFrame([list(f.root.vdmscan[0])], columns=f.root.vdmscan.colnames)
        #scan_info['ip'] = scan_info['ip'].apply(lambda ip: [i for i,b in enumerate(bin(ip)[::-1]) if b == '1']) # Binary to dec to list all scanning IPs
        scan_info['energy'] = f.root.scan5_beam[0]['egev']
    return scan_info[['timestampsec', 'fillnum', 'runnum', 'energy', 'ip', 'bstar5', 'xingHmurad']]


def get_beam_current_and_rates(filename,  scan, luminometers):
    with tables.open_file(filename, 'r') as f:
        collidable = np.nonzero(f.root.scan5_beam[0]['collidable'])[0] # indices of colliding bunches (0-indexed)
        filled = np.nonzero(np.logical_or(f.root.scan5_beam[0]['bxconfig1'], f.root.scan5_beam[0]['bxconfig2']))[0]

        rate_and_beam = pd.DataFrame()
        for p, plane in enumerate(scan.nominal_sep_plane.unique()):
            for b, sep in enumerate(scan[scan.nominal_sep_plane == plane].sep):
                print(scan)
                period_of_scanpoint = f'(timestampsec > {scan.min_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()}) & (timestampsec <= {scan.max_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()})'

                # Normalised beam current
                beam = np.array([b['bxintensity1'][collidable]*b['bxintensity2'][collidable]/1e22 for b in f.root['scan5_beam'].where(period_of_scanpoint)])

                # Mean over lumi sections
                new_data = pd.DataFrame(np.array([beam.mean(axis=0)]).T, columns=['beam'])
                #new_data = pd.DataFrame(np.array([r.mean(axis=0), stats.sem(r, axis=0), beam.mean(axis=0)]).T, columns=['rate', 'rate_err', 'beam'])

                for luminometer in luminometers:
                    # Rates of colliding bcids for this scan point
                    r = np.array([r['bxraw'][collidable] for r in f.root[luminometer].where(period_of_scanpoint)])
                    new_data[luminometer] = r.mean(axis=0)
                    new_data[f'{luminometer}_err'] = stats.sem(r, axis=0)

                new_data['fbct_dcct_fraction_b1'], new_data['fbct_dcct_fraction_b2'] = get_fbct_to_dcct_correction_factors(f, period_of_scanpoint, filled)

                new_data.insert(0, 'bcid', collidable+1) # Move to 1-indexed values of BCID
                new_data.insert(0, 'sep', sep)
                new_data.insert(0, 'plane', plane)

                rate_and_beam = new_data if p == 0 and b == 0 else pd.concat([rate_and_beam, new_data])

    for luminometer in luminometers:
        rate_and_beam[f'{luminometer}_normalised'] = rate_and_beam[luminometer] / rate_and_beam.beam
        rate_and_beam[f'{luminometer}_normalised_err'] = rate_and_beam[f'{luminometer}_err'] / rate_and_beam.beam

        # Add sensible error in case of 0 rate (max of error)
        rate_and_beam[f'{luminometer}_normalised_err'].replace(0, rate_and_beam[f'{luminometer}_normalised_err'].max(axis=0), inplace=True)

    #rate_and_beam.to_csv(f'{outpath}/rate_and_beam.csv', index=False)

    return rate_and_beam


def apply_beam_current_normalisation(rate_and_beam):
    # Mean over LS, multiply B1 * B2
    calib = rate_and_beam.groupby('plane')[['fbct_dcct_fraction_b1', 'fbct_dcct_fraction_b2']].transform('mean').prod(axis=1)
    rate_and_beam['beam_calibrated'] = rate_and_beam.beam / calib
    for rates in filter(lambda x: '_normalised' in x, rate_and_beam.columns):
        rate_and_beam[rates] *= calib
        rate_and_beam[rates] *= calib


def get_plot_template(filename, fill=None, energy=None):
    pdf = PdfPages(filename)

    fig = plt.figure()
    ax1 = fig.add_axes((.12,.3,.83,.65)) # Upper part: fit and data points

    hep.cms.label(llabel="Preliminary", rlabel=fr"Fill {fill}, $\sqrt{{s}}={energy:.1f}$ TeV", loc=1)

    ax1.set_ylabel('$R/(N_1 N_2)$ [arb.]')
    ax1.set_xticklabels([])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True, useOffset=False)
    ax1.minorticks_off()

    ax2 = fig.add_axes((.12,.1,.83,.2)) # Lower part: residuals
    ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax2.set_ylabel('Residual [$\sigma$]',fontsize=20)
    ax2.set_xlabel('$\Delta$ [mm]')
    ax2.minorticks_off()

    return pdf, ax1, ax2


def file_has_data(filename):
    with tables.open_file(filename, mode='r') as f:
        if not '/scan5_beam' in f or not '/vdmscan' in f:
            return False
        if np.array(f.root.scan5_beam).size < 10:
            return False
        return True

def main(args):
    if args.clean and os.path.isdir('output'):
        shutil.rmtree('output')

    filenames = sorted(list(filter(lambda x: file_has_data(x), args.files)))

    luminometers = args.luminometers.split(',')
    fits = args.fit.split(',')

    for fn, filename in enumerate(filenames):
        outpath = f'output/{Path(filename).stem}' # Save output to this folder
        Path(outpath).mkdir(parents=True, exist_ok=True) # Create output folder if not existing already

        scan_info = get_basic_info(filename)
        print(scan_info)
        scan = get_scan_info(filename)
        if scan.shape[0] < 5:  # less than 5 scan steps -> problems
            print(f'Found only {scan.shape[0]} scan steps, skipping')
            continue

        rate_and_beam = get_beam_current_and_rates(filename, scan, luminometers)

        if not args.no_fbct_dcct:
            apply_beam_current_normalisation(rate_and_beam)

        print(rate_and_beam)
        rate_and_beam.to_csv(f'{outpath}/data.csv', index=False)

        for f, fit in enumerate(fits):
            for l, luminometer in enumerate(luminometers):
                if args.pdf:
                    pdf, ax1, ax2 = get_plot_template(f'{outpath}/fit_{luminometer}_{fit}.pdf', scan_info.fillnum[0], scan_info['energy'][0]*2/1000)

                for p, plane in enumerate(rate_and_beam.plane.unique()):
                    for b, bcid in enumerate(rate_and_beam.bcid.unique()):
                        data_x = scan[scan.nominal_sep_plane == plane]['sep']
                        data_y = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid)][f'{luminometer}_normalised']
                        data_y_err = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid)][f'{luminometer}_normalised_err']

                        least_squares = LeastSquares(data_x, data_y, data_y_err, FIT_FUNCTIONS[fit]['handle']) # Initialise minimiser with data and fit function of choice
                        m = Minuit(least_squares, **FIT_FUNCTIONS[fit]['initial_values']) # Give the initial values defined in "FIT_FUNCTIONS"
                        for param, limit in PARAMETER_LIMITS.items():
                            m.limits[param] = limit
                        m.migrad()  # Finds minimum of least_squares function
                        m.hesse()   # Accurately computes uncertainties

                        new = pd.DataFrame([m.values], columns=m.parameters) # Store values and errors to dataframe
                        new = pd.concat([new, pd.DataFrame([m.errors], columns=m.parameters).add_suffix('_err')], axis=1) # Add suffix "_err" to errors
                        new['valid'] =  m.valid
                        new['accurate'] = m.accurate
                        new.insert(0, 'bcid', bcid)
                        new.insert(0, 'plane', plane)
                        new.insert(0, 'fit', fit)
                        new.insert(0, 'luminometer', luminometer)
                        new.insert(0, 'filename', Path(filename).stem)

                        fit_results = new if b == 0 and p == 0 else pd.concat([fit_results, new], ignore_index=True)

                        if args.pdf:
                            figure_items = []
                            figure_items.append(ax1.errorbar(data_x, data_y, data_y_err, fmt='ko')) # Plot the data points
                            x_dense = np.linspace(np.min(data_x), np.max(data_x))
                            figure_items.append(ax1.plot(x_dense, FIT_FUNCTIONS[fit]['handle'](x_dense, *m.values), 'k')) # Plot the fit result

                            fit_info = [f'{plane}, BCID {bcid}', f'$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}']
                            for param, v, e in zip(m.parameters, m.values, m.errors):
                                fit_info.append(f'{param} = ${v:.3e} \\pm {e:.3e}$')

                            fit_info = [info.replace('cap_sigma', '$\Sigma$') for info in fit_info]

                            figure_items.append(ax1.text(0.95, 0.95, '\n'.join(fit_info), transform=ax1.transAxes, fontsize=14, fontweight='bold',
                                verticalalignment='top', horizontalalignment='right'))

                            residuals = (data_y.to_numpy() - FIT_FUNCTIONS[fit]['handle'](data_x, *m.values).to_numpy()) / data_y_err.to_numpy()
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

                try:
                    val = fit_results.pivot(index='bcid', columns=['plane'], values=['cap_sigma', 'peak', 'cap_sigma_err', 'peak_err'])
                except Exception as e:
                    print(f'{filename}: {e}')
                    continue

                sigvis = np.pi * val.cap_sigma.prod(axis=1) * val.peak.sum(axis=1)

                sigvis_err = (val.cap_sigma_err**2 / val.cap_sigma**2).sum(axis=1) + (val.peak_err**2).sum(axis=1) / (val.peak).sum(axis=1)**2
                sigvis_err = np.sqrt(sigvis_err) * sigvis

                lumi = pd.concat([sigvis, sigvis_err], axis=1)
                lumi.columns = ['sigvis', 'sigvis_err']

                fit_results = new if b == 0 and p == 0 else pd.concat([fit_results, new], ignore_index=True)

                results = lumi.merge(fit_results, how='outer', on='bcid')
                all_results = results if fn == 0 and l == 0 and f == 0 else pd.concat([all_results, results], ignore_index=True)

        all_results.to_csv('output/result.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--luminometers', type=str, help='Luminometer name, separated by comma', required=True)
    parser.add_argument('-nofd', '--no_fbct_dcct', help='Do NOT calibrate beam current', action='store_true')
    parser.add_argument('-bkg', '--background_correction', help='Apply bckground correction', action='store_true')
    parser.add_argument('-pdf', '--pdf', help='Create fit PDFs', action='store_true')
    parser.add_argument('-fit', type=str, help=f'Fit function, give multiple by separating by comma. Choices: {FIT_FUNCTIONS.keys()}', default='sg')
    parser.add_argument('-c', '--clean', action='store_true', help='Make a clean output folder')
    parser.add_argument('files', nargs='*')

    main(parser.parse_args())

