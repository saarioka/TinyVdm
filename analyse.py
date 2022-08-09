import argparse
import shutil
import os
import multiprocessing
import traceback
from pathlib import Path
from functools import partial

import yaml
import tables
import pandas as pd
import numpy as np
from logging import debug, info, warning, error
from scipy import stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

import fits
import fit_plotter
import utilities as utl

pd.set_option('precision', 4)

CONFIG_FILENAME = 'config.yml'
with open(CONFIG_FILENAME, 'r') as cfg:
    CONFIG = yaml.safe_load(cfg)


def get_scan_info(filename):
    with tables.open_file(filename, 'r') as f:
        # Associate timestamps to scan plane - scan point -pairs

        quality_condition = '(stat == "ACQUIRING") & (nominal_sep_plane != "NONE")'
        scan = pd.DataFrame()
        scan['timestampsec'] = [r['timestampsec'] for r in f.root.vdmscan.where(quality_condition)]
        scan['sep'] = [r['sep'] for r in f.root.vdmscan.where(quality_condition)]

        # Decode is needed for values of type string
        scan['nominal_sep_plane'] = [r['nominal_sep_plane'].decode('utf-8') for r in f.root.vdmscan.where(quality_condition)]
        scan['nominal_sep_plane'] = scan['nominal_sep_plane'].astype(str)

        # Get min and max for each plane - sep pair
        scan = scan.groupby(['nominal_sep_plane', 'sep']).agg(min_time=('timestampsec', np.min), max_time=('timestampsec', np.max))

    return scan.reset_index()


def get_basic_info(filename):
    with tables.open_file(filename, 'r') as f:
        # Get first row of table "vdmscan" to save scan conditions that are constant thorugh the scan
        scan_info = pd.DataFrame([list(f.root.vdmscan[0])], columns=f.root.vdmscan.colnames)
        scan_info['time'] = pd.to_datetime(scan_info['timestampsec'], unit='s').dt.strftime('%d.%m.%Y, %H:%M:%S')
        #scan_info['ip'] = scan_info['ip'].apply(lambda ip: [i for i,b in enumerate(bin(ip)[::-1]) if b == '1']) # Binary to dec to list all scanning IPs
        scan_info['energy'] = f.root.scan5_beam[0]['egev']
    return scan_info[['time', 'fillnum', 'runnum', 'energy', 'ip', 'bstar5', 'xingHmurad']]


def get_beam_current_and_rates(filename,  scan, luminometers):
    with tables.open_file(filename, 'r') as f:
        collidable = np.nonzero(f.root.scan5_beam[0]['collidable'])[0]  # indices of colliding bunches (0-indexed)

        for index, row in scan.iterrows():
            period_of_scanpoint = f'(timestampsec > {row.min_time}) & (timestampsec <= {row.max_time})'

            # Normalised beam current
            beam = np.array([b['bxintensity1'][collidable]*b['bxintensity2'][collidable]/1e22 for b in f.root['scan5_beam'].where(period_of_scanpoint)])

            # Mean over lumi sections
            new_data = pd.DataFrame(np.array([beam.mean(axis=0)]).T, columns=['beam'])

            for luminometer in luminometers:
                # Rates of colliding bcids for this scan point
                r = np.array([r['bxraw'][collidable] for r in f.root[luminometer].where(period_of_scanpoint)])
                new_data[luminometer] = r.mean(axis=0)
                new_data[f'{luminometer}_err'] = stats.sem(r, axis=0)

            new_data['fbct_dcct_fraction_b1'], new_data['fbct_dcct_fraction_b2'] = get_fbct_to_dcct_correction_factors(f, period_of_scanpoint)

            new_data.insert(0, 'correction', 'none')
            new_data.insert(0, 'bcid', collidable + 1) # Move to 1-indexed values of BCID
            new_data.insert(0, 'sep', row.sep)
            new_data.insert(0, 'plane', row.nominal_sep_plane)

            rate_and_beam = new_data if index == 0 else pd.concat([rate_and_beam, new_data])

    return rate_and_beam.reset_index(drop=True)


def get_bkg_from_noncolliding(filename, rate_and_beam, luminometers):
    corrected_rate_and_beam = rate_and_beam.copy()
    corrected_rate_and_beam.correction = 'background'
    with tables.open_file(filename, 'r') as f:
        filled_noncolliding = np.nonzero(np.logical_xor(f.root.scan5_beam[0]['bxconfig1'], f.root.scan5_beam[0]['bxconfig2']))[0]
        for luminometer in luminometers:
            abort_gap_mask = [*range(3421, 3535)] if 'bcm1f' in luminometer else [*range(3444, 3564)]

            rate_nc = np.array([r['bxraw'][filled_noncolliding] for r in f.root[luminometer]])
            rate_ag = np.array([r['bxraw'][abort_gap_mask] for r in f.root[luminometer]])
            rate_nc = rate_nc[rate_nc >= 0]
            rate_ag = rate_ag[rate_ag >= 0]
            bkg = 2 * rate_nc.mean() - rate_ag.mean()
            bkg_err = np.sqrt(4*stats.sem(rate_nc)**2 + stats.sem(rate_ag)**2)
            info(f'background for {luminometer}:\t{bkg:.2e} +- {bkg_err:.2e} (RNC {rate_nc.mean():.2e}, RAG {rate_ag.mean():.2e})')

            rate_and_beam[f'bkg_{luminometer}'] = bkg
            rate_and_beam[f'bkg_{luminometer}_err'] = bkg_err
            rate_and_beam[luminometer] -= bkg
            rate_and_beam[f'{luminometer}_err'] = np.sqrt(rate_and_beam[f'{luminometer}_err']**2 + bkg_err**2)
    return corrected_rate_and_beam


def get_fbct_to_dcct_correction_factors(f, period_of_scanpoint):
    filled = np.nonzero(np.logical_or(f.root.scan5_beam[0]['bxconfig1'], f.root.scan5_beam[0]['bxconfig2']))[0]
    fbct_b1 = np.array([b['bxintensity1'][filled] for b in f.root['scan5_beam'].where(period_of_scanpoint)])
    fbct_b2 = np.array([b['bxintensity2'][filled] for b in f.root['scan5_beam'].where(period_of_scanpoint)])
    dcct = np.array([[b['intensity1'], b['intensity2']] for b in f.root['scan5_beam'].where(period_of_scanpoint)]) # Normalised beam current
    return np.array([fbct_b1.sum(), fbct_b2.sum()]) / dcct.sum(axis=0)


def apply_beam_current_normalisation(rate_and_beam):
    # Mean over LS, multiply B1 * B2
    calib = rate_and_beam.groupby('plane')[['fbct_dcct_fraction_b1', 'fbct_dcct_fraction_b2']].transform('mean').prod(axis=1)
    rate_and_beam['beam_calibrated'] = rate_and_beam.beam / calib
    for rates in filter(lambda c: '_normalised' in c, rate_and_beam.columns):
        rate_and_beam[rates] *= calib


def normalise_rates_current(rate_and_beam, luminometers):
    for luminometer in luminometers:
        rate_and_beam[f'{luminometer}_normalised'] = rate_and_beam[luminometer] / rate_and_beam.beam
        rate_and_beam[f'{luminometer}_normalised_err'] = rate_and_beam[f'{luminometer}_err'] / rate_and_beam.beam

        # Add sensible error in case of 0 rate (max of error)
        rate_and_beam[f'{luminometer}_normalised_err'].replace(0, rate_and_beam[f'{luminometer}_normalised_err'].max(axis=0), inplace=True)


def file_has_data(filename):
    with tables.open_file(filename, mode='r') as f:
        if '/scan5_beam' not in f or '/vdmscan' not in f:
            return False
        if np.array(f.root.scan5_beam).size < 10:
            return False
        if np.count_nonzero(f.root.scan5_beam[0]['collidable']) == 0:
            return False
        return True


def make_fit(x, y, yerr, fit):
    initial = CONFIG['fitting']['parameter_initial_values'][fit]

    if initial['peak'] == 'auto':
        initial['peak'] = np.max(y)

    least_squares = LeastSquares(x, y, yerr, fits.fit_functions[fit])  # Initialise minimiser with data and fit function of choice

    # Give the initial values defined in "fit_functions"
    m = Minuit(least_squares, **initial)

    for param, limit in CONFIG['fitting']['parameter_limits'].items():
        if param in m.parameters:
            m.limits[param] = limit

    m.migrad(ncall=99999, iterate=100)  # Finds minimum of least_squares function
    m.hesse()  # Uncertainties
    return m


def analyse(rate_and_beam, scan, pdf, filename, fill, energy, luminometer, correction, fit):
    if pdf:
        plotter = fit_plotter.plotter(f'output/fits/fit_{Path(filename).stem}_{utl.get_nice_name_for_luminometer(luminometer)}_{correction}_{fit}.pdf', fill, energy)
    try:
        for p, plane in enumerate(rate_and_beam.plane.unique()):
            for b, bcid in enumerate(rate_and_beam.bcid.unique()):
                data = rate_and_beam[(rate_and_beam.plane == plane) & (rate_and_beam.bcid == bcid) & (rate_and_beam.correction == correction)]
                x = scan[scan.nominal_sep_plane == plane]['sep']
                y = data[f'{luminometer}_normalised']
                yerr = data[f'{luminometer}_normalised_err']

                if fit == 'adaptive':
                    for used_fit, highest_order_param in [('dg', 'peak_ratio'), ('polyG6', 'r6'), ('polyG4', 'r4'), ('polyG2', 'r2'), ('sg', None)]:
                        m = make_fit(x, y, yerr, used_fit)
                        chi2 = m.fval / (len(x) - m.nfit)
                        if used_fit == 'sg':
                            break
                        limits = CONFIG['fitting']['adaptive']['parameters_significance_threshold'][highest_order_param]
                        if limits[0] < np.abs(m.values[highest_order_param]) < limits[1] \
                            and chi2 < float(CONFIG['fitting']['adaptive']['chi2_threshold']) \
                            and ((m.valid and CONFIG['fitting']['adaptive']['require_valid']) or not CONFIG['fitting']['adaptive']['require_valid']) \
                            and ((m.accurate and CONFIG['fitting']['adaptive']['require_accurate']) or not CONFIG['fitting']['adaptive']['require_accurate']):
                            break
                        debug('Plane %s, BCID %d, fit %s: parameter of merit %.2e, chi2 %.2e, valid %d, accurate %d',
                              plane, bcid, used_fit, np.abs(m.values[highest_order_param]), chi2, m.valid, m.accurate)

                    debug('Plane %s, BCID %d: using fit %s', plane, bcid, used_fit)
                else:
                    m = make_fit(x, y, yerr, fit)
                    used_fit = fit

                new = pd.DataFrame([m.values], columns=m.parameters) # Store values and errors to dataframe
                new = pd.concat([new, pd.DataFrame([m.errors], columns=m.parameters).add_suffix('_err')], axis=1) # Add suffix "_err" to errors

                new['valid'] = m.valid
                new['accurate'] = m.accurate
                new['chi2'] = m.fval / (len(x) - m.nfit)
                new.insert(0, 'bcid', bcid)
                new.insert(0, 'plane', plane)
                new.insert(0, 'fit', fit)
                new.insert(0, 'correction', correction)
                new.insert(0, 'luminometer', luminometer)
                new.insert(0, 'filename', Path(filename).stem)

                fit_results = new if b == 0 and p == 0 else pd.concat([fit_results, new], ignore_index=True)

                if pdf:
                    fit_info = [f'{utl.get_nice_name_for_luminometer(luminometer)}, {used_fit}',
                                correction,
                                f'{plane}, BCID {bcid}',
                                fr'$\chi^2$ / $n_\mathrm{{dof}}$ = {m.fval:.1f} / {len(x) - m.nfit}']
                    for param, v, e in zip(m.parameters, m.values, m.errors):
                        fit_info.append(f'{param} = ${fit_plotter.as_si(v):s} \\pm {fit_plotter.as_si(e):s}$')
                    fit_info.append(f'valid: {m.valid}, accurate: {m.accurate}')
                    fit_info = [info.replace('capsigma', r'$\Sigma$') for info in fit_info]
                    fit_info = '\n'.join(fit_info)

                    plotter.create_page(x, y, yerr, used_fit, fit_info, m.covariance, *m.values)

        if pdf:
            plotter.close_pdf()

        fit_results.capsigma *= 1e3  # to µm
        fit_results.capsigma_err *= 1e3  # to µm

        # Calculate sigvis
        val = fit_results.pivot(index='bcid', columns=['plane'])

        sigvis = np.pi * val.capsigma.prod(axis=1) * val.peak.sum(axis=1)

        sigvis_err = (val.capsigma_err**2 / val.capsigma**2).sum(axis=1) + (val.peak_err**2).sum(axis=1) / (val.peak).sum(axis=1)**2
        sigvis_err = np.sqrt(sigvis_err) * sigvis

        # Combine sigvis and fit results to a shared file
        plane_names = []
        [plane_names.append(c[1]) for c in val.columns if c[1] not in plane_names]

        plane_str_to_idx = dict(zip(plane_names, range(1, len(plane_names)+1)))

        val.columns = [f'{j}_{plane_str_to_idx[k]}' for j, k in val.columns] # pivottable -> table
        val.reset_index(inplace=True)

        # These are samee for both planes
        only_single = ['luminometer', 'fit', 'correction', 'filename']
        val.drop(columns=[s + '_2' for s in only_single], inplace=True)
        val.rename(columns=dict(zip([s + '_1' for s in only_single], only_single)), inplace=True)

        lumi = pd.concat([sigvis, sigvis_err], axis=1)
        lumi.columns = ['sigvis', 'sigvis_err']

        return lumi.merge(val, how='outer', on='bcid')

    except Exception:
        traceback.print_exc()
        return pd.DataFrame()


def main(args):
    utl.init_logger(args.verbosity)

    if not args.allow_cache and os.path.isdir('output'):
        shutil.rmtree('output')

    filenames = sorted(list(filter(file_has_data, args.files)))
    luminometers = args.luminometers.split(',')
    corrections = args.corrections.split(',')
    fitfunctions = args.fit.split(',')

    for folder in ('data', 'results', 'fits'):
        os.makedirs(f'output/{folder}', exist_ok=True)

    # Constant parameters are passed with partial and iterables as a list
    # Running for all luminometers in parallel.
    n_threads = min(len(luminometers) * len(corrections) * len(fitfunctions), CONFIG['runtime']['max_threads'] if CONFIG['runtime']['max_threads'].isdigit() else multiprocessing.cpu_count())
    with multiprocessing.Pool(n_threads) as pool:
        for fn, filename in enumerate(filenames):
            try:
                rate_and_beam_filename = f'output/data/data_{Path(filename).stem}.csv'

                scan_info = get_basic_info(filename)
                print(scan_info.to_string(index=False))
                scan = get_scan_info(filename)
                if scan.shape[0] < 10:  # less than 5 scan steps per scan -> problems
                    error(f'Found only {scan.shape[0]} scan steps (both planes total), skipping')
                    continue

                if Path(rate_and_beam_filename).is_file():
                    rate_and_beam = pd.read_csv(rate_and_beam_filename)
                    warning('Using cached data file')
                else:
                    rate_and_beam = get_beam_current_and_rates(filename, scan, luminometers)

                    if 'background' in corrections:
                        bkg = get_bkg_from_noncolliding(filename, rate_and_beam, luminometers)
                        rate_and_beam = pd.concat([rate_and_beam, bkg], axis=0, ignore_index=True)

                    normalise_rates_current(rate_and_beam, luminometers)

                    if not args.no_fbct_dcct:
                        apply_beam_current_normalisation(rate_and_beam)

                    rate_and_beam.to_csv(rate_and_beam_filename, index=False)

                jobs = []
                for l in luminometers:
                    for c in corrections:
                        for f in fitfunctions:
                            jobs.append((l, c, f))

                result = pool.starmap(func=partial(analyse, rate_and_beam, scan, args.pdf, filename, scan_info.fillnum[0], scan_info['energy'][0]*2/1000), iterable=jobs)
                result = pd.concat(result, ignore_index=True).reset_index(drop=True)
                result.to_csv(f'output/results/results_{Path(filename).stem}.csv')

                result_all = result if fn == 0 else pd.concat([result_all, result], ignore_index=True)

                if len(fitfunctions) > 1:
                    # print stats
                    for plane in (1, 2):
                        criteria = result.groupby(['correction', 'fit', 'bcid'])[[f'chi2_{plane}', f'capsigma_err_{plane}', f'peak_err_{plane}', f'mean_err_{plane}']].mean()
                        criteria.reset_index(inplace=True)
                        best_fits = criteria.loc[criteria.groupby(['correction', 'bcid'])[f'capsigma_err_{plane}'].idxmin()][['correction', 'bcid', 'fit']]
                        print(f'\nLowest error in capsigma, plane {plane}:')
                        print(best_fits.to_string(index=False))

                        best_fits = criteria.loc[criteria.groupby(['correction', 'bcid'])[f'chi2_{plane}'].idxmin()][['correction', 'bcid', 'fit']]
                        print(f'\nLowest chi2, plane {plane}:')
                        print(best_fits.to_string(index=False))

            except Exception:
                print(filename + ':')
                traceback.print_exc()
        result_all.to_csv('output/result.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--luminometers', type=str, help='Luminometer name, separated by comma', required=True)
    parser.add_argument('-nofd', '--no_fbct_dcct', help='Do NOT calibrate beam current', action='store_true')
    parser.add_argument('-c', '--corrections', type=str, help='Which corrections to apply (comma separated)', default='none')
    parser.add_argument('-pdf', '--pdf', help='Create fit PDFs', action='store_true')
    parser.add_argument('-fit', type=str, help='Fit function, give multiple by separating by comma', choices=list(fits.fit_functions.keys()).append('adaptive'), default='sg')
    parser.add_argument('-ac', '--allow_cache', action='store_true', help='Allow the use of cached data values (make sure to use the same arguments as before)')
    parser.add_argument('--verbosity', '-v', type=int, help='Verbosity level of printouts. Give a value between 1 and 5 (from least to most verbose)', choices=[1,2,3,4,5], default=4)
    parser.add_argument('files', nargs='*')

    main(parser.parse_args())
