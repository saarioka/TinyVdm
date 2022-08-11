"""
Calculate and apply corrections
"""
from logging import info, error
from pathlib import Path

import tables
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import utilities as utl

def get_bkg_from_noncolliding(filename, rate_and_beam, luminometers, plot=False):
    corrected_rate_and_beam = rate_and_beam.copy()
    corrected_rate_and_beam.correction = 'background'
    with tables.open_file(filename, 'r') as f:
        filled_noncolliding = np.nonzero(np.logical_xor(f.root.scan5_beam[0]['bxconfig1'], f.root.scan5_beam[0]['bxconfig2']))[0]
        if np.sum(filled_noncolliding) == 0:
            error('Background correction specified, but no non-colliding bunches present')
            return None
        for luminometer in luminometers:
            abort_gap_mask = [*range(3421, 3535)] if 'bcm1f' in luminometer else [*range(3444, 3564)]

            rate_nc = utl.from_h5(f, luminometer, 'bxraw', mask=filled_noncolliding)
            rate_ag = utl.from_h5(f, luminometer, 'bxraw', mask=abort_gap_mask)

            if plot:
                fig, ax = plt.subplots()
                ax.scatter(filled_noncolliding, rate_nc.mean(axis=0), 75, label='Non-colliding')
                ax.scatter(abort_gap_mask, rate_ag.mean(axis=0), 75, label='Abort gap')
                ax.set_xlabel('BCID')
                ax.set_ylabel('Rate')
                fig.suptitle(f'Background, {utl.get_nice_name_for_luminometer(luminometer)}')
                fig.legend()
                fig.tight_layout()
                fig.savefig(f'output/figures/background/bkg_{Path(filename).stem}_{utl.get_nice_name_for_luminometer(luminometer)}.pdf')

            rate_nc = rate_nc[rate_nc >= 0]
            rate_ag = rate_ag[rate_ag >= 0]

            bkg = 2 * rate_nc.mean() - rate_ag.mean()
            bkg_err = np.sqrt(4*stats.sem(rate_nc)**2 + stats.sem(rate_ag)**2)

            print(f'Background for {utl.get_nice_name_for_luminometer(luminometer): <10} {bkg:.2e} +- {bkg_err:.2e} (RNC {rate_nc.mean():.2e}, RAG {rate_ag.mean():.2e})')

            rate_and_beam[f'bkg_{luminometer}'] = bkg
            rate_and_beam[f'bkg_{luminometer}_err'] = bkg_err
            rate_and_beam[luminometer] -= bkg
            rate_and_beam[f'{luminometer}_err'] = np.sqrt(rate_and_beam[f'{luminometer}_err']**2 + bkg_err**2)
    return corrected_rate_and_beam


def get_fbct_to_dcct_correction_factors(f, period_of_scanpoint):
    filled = np.nonzero(np.logical_or(f.root.scan5_beam[0]['bxconfig1'], f.root.scan5_beam[0]['bxconfig2']))[0]
    fbct_b1 = utl.from_h5(f, 'scan5_beam', 'bxintensity1', period_of_scanpoint, filled)
    fbct_b2 = utl.from_h5(f, 'scan5_beam', 'bxintensity2', period_of_scanpoint, filled)
    dcct = np.array([[b['intensity1'], b['intensity2']] for b in f.root['scan5_beam'].where(period_of_scanpoint)]) # Normalised beam current
    return np.array([fbct_b1.sum(), fbct_b2.sum()]) / dcct.sum(axis=0)


def apply_beam_current_normalisation(rate_and_beam):
    # Mean over LS, multiply B1 * B2
    calib = rate_and_beam.groupby('plane')[['fbct_dcct_fraction_b1', 'fbct_dcct_fraction_b2']].transform('mean').prod(axis=1)
    rate_and_beam['beam_calibrated'] = rate_and_beam.beam / calib
    for rates in filter(lambda c: '_normalised' in c, rate_and_beam.columns):
        rate_and_beam[rates] *= calib


def apply_rate_normalisation(rate_and_beam, luminometers):
    for luminometer in luminometers:
        rate_and_beam[f'{luminometer}_normalised'] = rate_and_beam[luminometer] / rate_and_beam.beam
        rate_and_beam[f'{luminometer}_normalised_err'] = rate_and_beam[f'{luminometer}_err'] / rate_and_beam.beam

        # Add sensible error in case of 0 rate (max of error)
        rate_and_beam[f'{luminometer}_normalised_err'].replace(0, rate_and_beam[f'{luminometer}_normalised_err'].max(axis=0), inplace=True)


