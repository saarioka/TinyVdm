from pathlib import Path
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
import tables
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

matplotlib.use('Agg')

plt.style.use(hep.style.CMS)

# Define functions that may be fit to the rates
def sg(x, peak, mean, cap_sigma):
    return peak*np.exp(-(x-mean)**2/(2*cap_sigma**2))

def sg_const(x, peak, mean, cap_sigma, constant):
    return sg(x, peak, mean, cap_sigma) + constant

def dg(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio):
    return sg(x, peak*peak_ratio, mean, cap_sigma*cap_sigma_ratio) + sg(x, peak*(1-peak_ratio), mean, cap_sigma*(1-cap_sigma_ratio))

def dg_const(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio, constant):
    return dg(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio) + constant

# Each function needs a mapping from string given as a parameter, and also a set of initial conditions. Initial value for peak can be estimated well as normalised head-on rate
FIT_FUNCTIONS = {
    'sg':       {'handle': sg,       'initial_values': {'mean': 0, 'cap_sigma': 0.25}},
    'sg_const': {'handle': sg_const, 'initial_values': {'mean': 0, 'cap_sigma': 0.25, 'constant': 0}},
    'dg':       {'handle': dg,       'initial_values': {'mean': 0, 'cap_sigma': 0.25, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5}},
    'dg_const': {'handle': dg_const, 'initial_values': {'mean': 0, 'cap_sigma': 0.25, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5, 'constant': 0}}
}

def bkg_from_noncolliding(f, period_of_scanpoint, luminometer): # WIP
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
            general_info = pd.DataFrame([list(f.root.vdmscan[0])], columns=f.root.vdmscan.colnames) # Get first row of table "vdmscan" to save scan conditions that are constant through the scan
            general_info['ip'] = general_info['ip'].apply(lambda ip: [i for i,b in enumerate(bin(ip)[::-1]) if b == '1']) # Binary to dec to list all scanning IPs
            general_info['energy'] = f.root.beam[0]['egev']
            general_info[['fillnum', 'runnum', 'timestampsec', 'energy', 'ip', 'bstar5', 'xingHmurad']].to_csv(f'{outpath}/scan.csv', index=False) # Save this set of conditions to file

            collidable = np.nonzero(f.root.beam[0]['collidable'])[0] # indices of colliding bunches (0-indexed)
            filled = np.nonzero(np.logical_or(f.root.beam[0]['bxconfig1'], f.root.beam[0]['bxconfig2']))[0]

            # Associate timestamps to scan plane - scan point -pairs
            scan = pd.DataFrame()

            scan['timestampsec'] = [r['timestampsec'] for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
            scan['sep'] = [r['sep'] for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
            scan['nominal_sep_plane'] = [r['nominal_sep_plane'].decode('utf-8') for r in f.root.vdmscan.where('stat == "ACQUIRING"')] # Decode is needed for values of type string
            scan = scan.groupby(['nominal_sep_plane', 'sep']).agg(min_time=('timestampsec', np.min), max_time=('timestampsec', np.max)) # Get min and max for each plane - sep pair
            scan.reset_index(inplace=True)
            scan.groupby(['nominal_sep_plane', 'sep'])

            data = pd.DataFrame()
            for p, plane in tqdm(enumerate(scan.nominal_sep_plane.unique())):
                for b, sep in tqdm(enumerate(scan.sep.unique()), leave=False):
                    new = pd.DataFrame()
                    new['bcid'] = range(1,3565)

                    period_of_scanpoint = f'(timestampsec > {scan.min_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()}) & (timestampsec <= {scan.max_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()})'

                    r = np.array([r['bxraw'] for r in f.root[args.luminometer].where(period_of_scanpoint)])
                    new['rate'] = r.mean(axis=0)
                    new['rate_err'] = stats.sem(r, axis=0)

                    fbct1 = np.array([b['bxintensity1'] for b in f.root['beam'].where(period_of_scanpoint)])
                    fbct2 = np.array([b['bxintensity2'] for b in f.root['beam'].where(period_of_scanpoint)])
                    new['fbct1'] = fbct1.mean(axis=0)
                    new['fbct2'] = fbct2.mean(axis=0)
                    new['fbct1_err'] = stats.sem(fbct1, axis=0)
                    new['fbct2_err'] = stats.sem(fbct2, axis=0)

                    dcct1 = np.array([b['intensity1'] for b in f.root['beam'].where(period_of_scanpoint)])
                    dcct2 = np.array([b['intensity2'] for b in f.root['beam'].where(period_of_scanpoint)])
                    new['dcct1'] = dcct1.mean(axis=0)
                    new['dcct2'] = dcct2.mean(axis=0)
                    new['dcct1_err'] = stats.sem(dcct1, axis=0)
                    new['dcct2_err'] = stats.sem(dcct2, axis=0)

                    new['filled'] = np.logical_or(f.root.beam[0]['bxconfig1'], f.root.beam[0]['bxconfig2'])
                    new['colliding'] = np.logical_and(f.root.beam[0]['bxconfig1'], f.root.beam[0]['bxconfig2'])
                    new['filled_noncolliding'] = np.logical_xor(f.root.beam[0]['bxconfig1'], f.root.beam[0]['bxconfig2'])

                    # DCCT is not per bunch, instead same value that contains te sum over BCIDs is repeated for all BCIDs
                    new['fbct_to_dcct1'], new['fbct_to_dcct2'] = new.query('filled')[['fbct1', 'fbct2']].sum(axis=0).to_numpy() / new.query('bcid == 1')[['dcct1', 'dcct2']].sum(axis=0).to_numpy()

                    new.insert(0, 'sep', sep)
                    new.insert(0, 'plane', plane)

                    data = pd.concat([data, new])

                    #bkg = bkg_from_noncolliding(f, period_of_scanpoint, args.luminometer)

        data.reset_index(drop=True, inplace=True)

        beam = data['fbct1'][filled] * data['fbct2'][filled] / 1e22
        print(data['fbct1'][filled].sum())
        print(data['fbct2'][filled].sum())
        data['rate_normalised'] = data.rate / beam
        data['rate_normalised_err'] = data.rate_err / beam

        if args.calibrate_beam_current:
            calib = data.groupby('plane')[['fbct_to_dcct1', 'fbct_to_dcct2']].mean().prod(axis=1) # mean over LS, prod over beams
            for p in calib.index.unique():
                data.loc[data.plane == p, 'rate_normalised'] *= calib[calib.index == p].item()
                data.loc[data.plane == p, 'rate_normalised_err'] *= calib[calib.index == p].item()

        data.replace([-np.inf, np.inf, np.nan], 0, inplace=True)

        # Add sensible error in case of 0 rate: max of error, most often error associated to point with 1 hit (if there is a point with 0 rate, then there is probably one with 1)
        data['rate_normalised_err'].replace(0, data['rate_normalised_err'].max(), inplace=True)

        data.to_csv(f'{outpath}/{args.luminometer}_data.csv', index=False)

        if args.pdf: # initialize template for plots
            pdf = PdfPages(f'{outpath}/fit_{args.luminometer}.pdf')

            fig = plt.figure()
            ax1 = fig.add_axes((.12,.3,.83,.65)) # Upper part: fit and data points
            hep.cms.label(llabel="Preliminary", rlabel=fr"Fill {general_info.fillnum[0]}, $\sqrt{{s}}={general_info['energy'][0]*2/1000:.1f}$ TeV", loc=1)
            ax1.set_ylabel('$R/(N_1 N_2)$ [arb.]')
            ax1.set_xticklabels([])
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True, useOffset=False)
            ax1.minorticks_off()

            ax2 = fig.add_axes((.12,.1,.83,.2)) # Lower part: residuals
            ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax2.set_ylabel(r'Residual [$\sigma$]',fontsize=20)
            ax2.set_xlabel(r'$\Delta$ [mm]')
            ax2.minorticks_off()

        for p, plane in enumerate(data.plane.unique()): # For each plane
            for b, bcid in enumerate(collidable+1): # For each BCID
                data_x = scan[scan.nominal_sep_plane == plane]['sep'] # x-data: separations
                data_y = data[(data.plane == plane) & (data.bcid == bcid)]['rate_normalised'] # y-data: normalised rates
                data_y_err = data[(data.plane == plane) & (data.bcid == bcid)]['rate_normalised_err']

                FIT_FUNCTIONS[args.fit]['initial_values']['peak'] = np.max(data_y)

                least_squares = LeastSquares(data_x, data_y, data_y_err, FIT_FUNCTIONS[args.fit]['handle']) # Initialise minimiser with data and fit function of choice
                m = Minuit(least_squares, **FIT_FUNCTIONS[args.fit]['initial_values']) # Give the initial values defined in "FIT_FUNCTIONS"
                m.limits['cap_sigma'] = [0, None] # Most preferably positive beam widths
                m.migrad(iterate=1)  # Finds minimum of least_squares function, iterate=1 disables max calls
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
                    figure_items.append(ax1.errorbar(data_x, data_y, data_y_err, fmt='ko', capsize=5)) # Plot the data points
                    x_dense = np.linspace(np.min(data_x), np.max(data_x))
                    figure_items.append(ax1.plot(x_dense, FIT_FUNCTIONS[args.fit]['handle'](x_dense, *m.values), 'k')) # Plot the fit result

                    fit_info = [f'{plane}, BCID {bcid}', f'$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}']
                    for param, v, e in zip(m.parameters, m.values, m.errors):
                        fit_info.append(f'{param} = ${v:.3e} \\pm {e:.3e}$')

                    fit_info = [info.replace('cap_sigma', r'$\Sigma$') for info in fit_info]

                    figure_items.append(ax1.text(0.95, 0.95, '\n'.join(fit_info), transform=ax1.transAxes, fontsize=14, fontweight='bold',
                        verticalalignment='top', horizontalalignment='right'))

                    residuals = (data_y.to_numpy() - FIT_FUNCTIONS[args.fit]['handle'](data_x, *m.values).to_numpy()) / data_y_err.to_numpy()
                    figure_items.append(ax2.scatter(data_x, residuals, c='k'))

                    # plot without changing xlim
                    lim = list(plt.xlim())
                    figure_items.append(ax2.plot(lim, [0, 0], 'k:'))
                    plt.xlim(lim)

                    pdf.savefig()

                    for item in figure_items: # Only delete lines and fit results, leave general things
                        if isinstance(item, list):
                            item[0].remove()
                        else:
                            item.remove()
        if args.pdf:
            pdf.close()

        fit_results.cap_sigma *= 1e3 # to µm
        fit_results.to_csv(f'{outpath}/{args.luminometer}_fit_results.csv', index=False)

        val = fit_results.pivot(index='bcid', columns=['plane'], values=['cap_sigma', 'peak', 'cap_sigma_err', 'peak_err'])

        sigvis = np.pi * val.cap_sigma.prod(axis=1) * val.peak.sum(axis=1)

        sigvis_err = (val.cap_sigma_err**2 / val.cap_sigma**2).sum(axis=1) + (val.peak_err**2).sum(axis=1) / (val.peak).sum(axis=1)**2
        sigvis_err = np.sqrt(sigvis_err) * sigvis

        lumi = pd.concat([sigvis, sigvis_err], axis=1)
        lumi.columns = ['sigvis', 'sigvis_err']
        lumi.to_csv(f'{outpath}/{args.luminometer}_lumi.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--luminometer', type=str, help='Luminometer name', required=True)
    parser.add_argument('-cbc', '--calibrate_beam_current', help='Calibrate beam current', action='store_true')
    parser.add_argument('-bkg', '--background_correction', help='Apply bckground correction', action='store_true')
    parser.add_argument('-pdf', '--pdf', help='Create fit PDFs', action='store_true')
    parser.add_argument('--fit', type=str, help='Fit function', choices=FIT_FUNCTIONS.keys(), default='sg')
    parser.add_argument('files', nargs='*')
    main(parser.parse_args())
