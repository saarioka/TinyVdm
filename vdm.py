import pandas as pd
import numpy as np
import tables
import matplotlib.pyplot as plt
from scipy import stats
from iminuit import Minuit
from iminuit.cost import LeastSquares
import mplhep as hep

plt.style.use(hep.style.CMS)

def gaussian(x, peak, mean, cap_sigma):
    return peak*np.exp(-(x-mean)**2/(2*cap_sigma**2))

data = pd.DataFrame()
with tables.open_file('7525_2110302352_2110310014.hd5', 'r') as f:
    data['timestampsec'] = [r['timestampsec'] for r in f.root.beam.where('timestampsec > 0')]

    # beam
    collidable = np.nonzero(f.root.beam[0]['collidable'])[0] # indices of colliding
    #bxintensity1 = np.array([r['bxintensity1'][collidable] for r in f.root.beam.where('timestampsec > 0')])
    #bxintensity2 = np.array([r['bxintensity2'][collidable] for r in f.root.beam.where('timestampsec > 0')])

    data['intensity1'] = [r['intensity1'] for r in f.root.beam.where('timestampsec > 0')]
    data['intensity2'] = [r['intensity2'] for r in f.root.beam.where('timestampsec > 0')]

    # scan
    scan = pd.DataFrame()
    scan['timestampsec'] = [r['timestampsec'] for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
    scan['sep'] = [r['sep'] for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
    scan['nominal_sep_plane'] = [r['nominal_sep_plane'].decode('utf-8') for r in f.root.vdmscan.where('stat == "ACQUIRING"')]
    scan = scan.groupby(['nominal_sep_plane', 'sep']).agg(min_time=('timestampsec', np.min), max_time=('timestampsec', np.max))

    scan.reset_index(inplace=True)
    #print(scan)

    #rate = pd.DataFrame()

    # rate
    for j, plane in enumerate(scan.nominal_sep_plane.unique()):
        rate = np.empty([len(scan[scan.nominal_sep_plane == plane]), len(collidable)])
        rate_err = np.empty([len(scan[scan.nominal_sep_plane == plane]), len(collidable)])
        for i, sep in enumerate(scan.sep.unique()):
            period_of_scanpoint = f"(timestampsec > {scan.min_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()}) & (timestampsec <= {scan.max_time[(scan.nominal_sep_plane == plane) & (scan.sep == sep)].item()})"
            r = np.array([r['bxraw'][collidable] for r in f.root['pltlumizero'].where(period_of_scanpoint)])
            b = np.array([b['bxintensity1'][collidable]*b['bxintensity2'][collidable]/1e22 for b in f.root['beam'].where(period_of_scanpoint)])
            #b = np.array([b['intensity1']*b['intensity2']/1e22 for b in f.root['beam'].where(period_of_scanpoint)])
            b_mean = b.mean(axis=0)
            rate[i,:] = r.mean(axis=0) / b_mean
            #rate[i,:] = r.mean(axis=0) # normal ratefile
            rate_err[i,:] = stats.sem(r, axis=0) / b_mean

        pd.DataFrame(rate, columns=collidable).to_csv('ratefile.csv', index=False)


        for i, bcid in enumerate(collidable):
            plt.figure()
            plt.title(f"{plane}, BCID {bcid+1}")
            data_x = scan[scan.nominal_sep_plane == plane]["sep"]
            data_y = rate[:, i]
            data_yerr = rate_err[:, i]

            data_yerr[data_yerr == 0] = np.mean(data_yerr)
            print(data_yerr)

            #rate_table = pd.DataFrame(np.array([data_x, rate[:,i], rate_err[:,i]]).T, columns=['sep', 'rate', 'err'])
            #rate_table = rate_table[(rate_table != 0).all(1)]

            #data_x = rate_table.sep
            #data_y = rate_table.rate
            #data_yerr = rate_table.err

            least_squares = LeastSquares(data_x, data_y, data_yerr, gaussian)

            m = Minuit(least_squares, peak=1e-4, mean=0, cap_sigma=0.3)

            m.migrad()  # finds minimum of least_squares function
            m.hesse()   # accurately computes uncertainties
            #print(m.values)

            new = pd.DataFrame([m.values, m.errors], columns=m.parameters)
            new.insert(0, 'what', ['value', 'error'])
            new.insert(0, 'plane', plane)
            new.insert(0, 'bcid', bcid)

            fit_results = new if i == 0 and j == 0 else pd.concat([fit_results, new], ignore_index=True)

            plt.errorbar(data_x, data_y, data_yerr, fmt="o")
            plt.plot(data_x, gaussian(data_x, *m.values))

            fit_info = [f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}"]

            for p, v, e in zip(m.parameters, m.values, m.errors):
                fit_info.append(f"{p} = ${v:.3e} \\pm {e:.3e}$")

            plt.legend(title="\n".join(fit_info))

            plt.yscale('log')

            plt.savefig(f'fit_{plane}_{bcid}.png')

    fit_results.cap_sigma *= 1e3

    values = fit_results[fit_results.what == 'value']
    errors = fit_results[fit_results.what == 'error']

    xsec = np.pi * values.groupby('bcid').cap_sigma.prod() * values.groupby('bcid').peak.sum()

    print(xsec)
