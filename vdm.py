from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import tables
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

def sg(x, peak, mean, cap_sigma):
    return peak*np.exp(-(x-mean)**2/(2*cap_sigma**2))

def sg_const(x, peak, mean, cap_sigma, constant):
    return sg(x, peak, mean, cap_sigma) + constant

def dg(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio):
    return sg(x, peak*peak_ratio, mean, cap_sigma*cap_sigma_ratio) + sg(x, peak*(1-peak_ratio), mean, cap_sigma*(1-cap_sigma_ratio))

def dg_const(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio, constant):
    return dg(x, peak, mean, cap_sigma, peak_ratio, cap_sigma_ratio) + constant

FIT_FUNCTIONS = {
    'sg':       {'handle': sg,       'initial_values': {'peak': 1e-4, 'mean': 0, 'cap_sigma': 0.3}},
    'sg_const': {'handle': sg_const, 'initial_values': {'peak': 1e-4, 'mean': 0, 'cap_sigma': 0.3, 'constant': 0}},
    'dg':       {'handle': dg,       'initial_values': {'peak': 2e-4, 'mean': 0, 'cap_sigma': 0.4, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5}},
    'dg_const': {'handle': dg_const, 'initial_values': {'peak': 2e-4, 'mean': 0, 'cap_sigma': 0.4, 'peak_ratio': 0.5, 'cap_sigma_ratio': 0.5, 'constant': 0}}
}

def main(args):
    for filename in args.files:
        outpath = f'output/{Path(filename).stem}'
        Path(outpath).mkdir(parents=True, exist_ok=True)
        with tables.open_file(filename, 'r') as f:
            collidable = np.nonzero(f.root.beam[0]['collidable'])[0] # indices of colliding bunches

            # Associate timestamps to scan plane - scan point -pairs
            scan = pd.DataFrame()
            scan['timestampsec'] = [r['timestampsec'] for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
            scan['sep'] = [r['sep'] for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
            scan['nominal_sep_plane'] = [r['nominal_sep_plane'].decode('utf-8') for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
            scan = scan.groupby(['nominal_sep_plane', 'sep']).agg(min_time=('timestampsec', np.min), max_time=('timestampsec', np.max))
            scan.reset_index(inplace=True)
            scan_file = pd.DataFrame([list(f.root.vdmscan[0])], columns=f.root.vdmscan.colnames)
            scan_file['ip'] = scan_file['ip'].apply(lambda ip: [i for i,b in enumerate(bin(ip)[::-1]) if b == '1']) # binary to dec to list all ips
            scan_file[['fillnum', 'runnum', 'timestampsec', 'ip', 'bstar5', 'xingHmurad']].to_csv(f'{outpath}/scan.csv', index=False)

            rate_and_beam = pd.DataFrame()
            for p, plane in enumerate(scan.nominal_sep_plane.unique()):
                for b, sep in enumerate(scan.sep.unique()):
                    period_of_scanpoint = f'(timestampsec > {scan.min_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()}) & (timestampsec <= {scan.max_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()})'

                    r = np.array([r['bxraw'][collidable] for r in f.root[args.luminometer].where(period_of_scanpoint)])
                    beam = np.array([b['bxintensity1'][collidable]*b['bxintensity2'][collidable]/1e22 for b in f.root['beam'].where(period_of_scanpoint)])
                    new_data = pd.DataFrame(np.array([r.mean(axis=0), beam.mean(axis=0)]).T, columns=['rate', 'beam'])

                    new_data.insert(0, 'bcid', collidable+1)
                    new_data.insert(0, 'sep', sep)
                    new_data.insert(0, 'plane', plane)

                    new_data['rate_normalised'] = new_data['rate'] / new_data['beam']
                    new_data['rate_normalised_err'] = stats.sem(r, axis=0) / new_data['beam']

                    rate_and_beam = new_data if p == 0 and b == 0 else pd.concat([rate_and_beam, new_data])

        rate_and_beam.to_csv(f'{outpath}/rate_and_beam.csv', index=False)

        rate_and_beam['rate_normalised_err'].replace(0, rate_and_beam['rate_normalised_err'].max(axis=0), inplace=True) # Add sensible error in case of 0 rate

        with PdfPages(f'{outpath}/fit_{args.luminometer}.pdf') as pdf:
            for p, plane in enumerate(rate_and_beam.plane.unique()):
                for b, bcid in enumerate(rate_and_beam.bcid.unique()):
                    plt.figure()
                    plt.title(f'{plane}, BCID {bcid+1}')
                    data_x = scan[scan.nominal_sep_plane == plane]['sep']
                    data_y = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid)]['rate_normalised']
                    data_y_err = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid)]['rate_normalised_err']

                    least_squares = LeastSquares(data_x, data_y, data_y_err, FIT_FUNCTIONS[args.fit]['handle'])
                    m = Minuit(least_squares, **FIT_FUNCTIONS[args.fit]['initial_values'])
                    m.migrad()  # finds minimum of least_squares function
                    m.hesse()   # accurately computes uncertainties

                    new = pd.DataFrame([m.values, m.errors], columns=m.parameters)
                    new.insert(0, 'what', ['value', 'error'])
                    new.insert(0, 'plane', plane)
                    new.insert(0, 'bcid', bcid)

                    fit_results = new if b == 0 and p == 0 else pd.concat([fit_results, new], ignore_index=True)

                    plt.errorbar(data_x, data_y, data_y_err, fmt='ko')
                    x_dense = np.linspace(np.min(data_x), np.max(data_x))
                    plt.plot(x_dense, FIT_FUNCTIONS[args.fit]['handle'](x_dense, *m.values), 'k')

                    fit_info = [f'$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}']
                    for param, v, e in zip(m.parameters, m.values, m.errors):
                        fit_info.append(f'{param} = ${v:.3e} \\pm {e:.3e}$')

                    plt.legend(title='\n'.join(fit_info))
                    pdf.savefig()

        fit_results.cap_sigma *= 1e3 # to µm

        values = fit_results[fit_results.what == 'value']
        errors = fit_results[fit_results.what == 'error']

        val = values.pivot(index='bcid', columns=['plane'], values=['cap_sigma', 'peak'])
        err = errors.pivot(index='bcid', columns=['plane'], values=['cap_sigma', 'peak'])

        sigvis = np.pi * val.cap_sigma.prod(axis=1) * val.peak.sum(axis=1)

        sigvis_err = (err.cap_sigma**2 / val.cap_sigma**2).sum(axis=1) + (err.peak**2).sum(axis=1) / (val.peak).sum(axis=1)**2
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

