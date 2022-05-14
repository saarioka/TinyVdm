from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import tables
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

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

# Each function needs a mapping from string given as a parameter, and also a set of initial conditions
FIT_FUNCTIONS = {
    'sg':       {'handle': sg,       'initial_values': {'peak': 1e-4, 'mean': 0, 'cap_sigma': 0.3}},
    'sg_const': {'handle': sg_const, 'initial_values': {'peak': 1e-4, 'mean': 0, 'cap_sigma': 0.3, 'constant': 0}},
    'dg':       {'handle': dg,       'initial_values': {'peak': 2e-4, 'mean': 0, 'cap_sigma': 0.4, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5}},
    'dg_const': {'handle': dg_const, 'initial_values': {'peak': 2e-4, 'mean': 0, 'cap_sigma': 0.4, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5, 'constant': 0}}
}

def main(args):
    for filename in args.files:
        outpath = f'output/{Path(filename).stem}' # Save output to this folder
        Path(outpath).mkdir(parents=True, exist_ok=True) # Create output folder if not existing already
        with tables.open_file(filename, 'r') as f:
            collidable = np.nonzero(f.root.beam[0]['collidable'])[0] # indices of colliding bunches (0-indexed)

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
                    new_data = pd.DataFrame(np.array([r.mean(axis=0), beam.mean(axis=0)]).T, columns=['rate', 'beam']) # Mean over lumi sections

                    new_data.insert(0, 'bcid', collidable+1) # Move to 1-indexed values of BCID
                    new_data.insert(0, 'sep', sep)
                    new_data.insert(0, 'plane', plane)

                    new_data['rate_normalised'] = new_data['rate'] / new_data['beam'] # Calculate normalised rate and rate error
                    new_data['rate_normalised_err'] = stats.sem(r, axis=0) / new_data['beam']

                    rate_and_beam = new_data if p == 0 and b == 0 else pd.concat([rate_and_beam, new_data])

        rate_and_beam.to_csv(f'{outpath}/rate_and_beam.csv', index=False)

        rate_and_beam['rate_normalised_err'].replace(0, rate_and_beam['rate_normalised_err'].max(axis=0), inplace=True) # Add sensible error in case of 0 rate (max of error)

        with PdfPages(f'{outpath}/fit_{args.luminometer}.pdf') as pdf:
            for p, plane in enumerate(rate_and_beam.plane.unique()): # For eah plane
                for b, bcid in enumerate(rate_and_beam.bcid.unique()): # For each BCID
                    #fig, ax = plt.subplots()
                    fig = plt.figure()
                    frame1 = fig.add_axes((.1,.3,.8,.6)) # Upper part
                    data_x = scan[scan.nominal_sep_plane == plane]['sep'] # x-data: separations
                    data_y = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid)]['rate_normalised'] # y-data: normalised rates
                    data_y_err = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid)]['rate_normalised_err']

                    least_squares = LeastSquares(data_x, data_y, data_y_err, FIT_FUNCTIONS[args.fit]['handle']) # Initialise minimiser with data and fit function of choice
                    m = Minuit(least_squares, **FIT_FUNCTIONS[args.fit]['initial_values']) # Give the initial values defined in "FIT_FUNCTIONS"
                    m.migrad()  # Finds minimum of least_squares function
                    m.hesse()   # Accurately computes uncertainties

                    new = pd.DataFrame([m.values], columns=m.parameters) # Store values and errors to dataframe
                    new = pd.concat([new, pd.DataFrame([m.errors], columns=m.parameters).add_suffix('_err')], axis=1) # Add suffix "_err" to errors
                    new.insert(0, 'bcid', bcid)
                    new.insert(0, 'plane', plane)

                    fit_results = new if b == 0 and p == 0 else pd.concat([fit_results, new], ignore_index=True)

                    plt.errorbar(data_x, data_y, data_y_err, fmt='ko') # Plot the data points
                    x_dense = np.linspace(np.min(data_x), np.max(data_x))
                    plt.plot(x_dense, FIT_FUNCTIONS[args.fit]['handle'](x_dense, *m.values), 'k') # Plot the fit result
                    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True, useOffset=False)
                    hep.cms.label(llabel="Preliminary", rlabel=fr"Fill {scan_info.fillnum[0]}, $\sqrt{{s}}={scan_info['energy'][0]*2/1000:.1f}$ TeV", loc=1)

                    fit_info = [f'{plane}, BCID {bcid}', f'$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}']
                    for param, v, e in zip(m.parameters, m.values, m.errors):
                        fit_info.append(f'{param} = ${v:.3e} \\pm {e:.3e}$')

                    plt.text(0.95, 0.95, '\n'.join(fit_info), transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')
                    plt.xlabel('$\Delta$ [mm]')
                    frame1.set_ylabel('$R/(N_1 N_2)$ [arb.]')
                    frame1.set_xticklabels([])

                    frame2 = fig.add_axes((.1,.1,.8,.2))
                    plt.ticklabel_format(axis='y', style='plain', useMathText=True, useOffset=False)
                    frame2.set_ylabel('Residual [$\sigma$]',fontsize=20)
                    residuals = (data_y.to_numpy() - FIT_FUNCTIONS[args.fit]['handle'](data_x, *m.values).to_numpy()) / data_y_err.to_numpy()
                    plt.scatter(data_x, residuals, c='k')

                    pdf.savefig(bbox_inches='tight')

        fit_results.cap_sigma *= 1e3 # to µm
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
    parser.add_argument('--fit', type=str, help='Fit function', choices=FIT_FUNCTIONS.keys(), default='sg')
    parser.add_argument('files', nargs='*')
    main(parser.parse_args())

