# README

Lightweight vdM analysis meant to illustrate the process of the vdM framework.

Will run with a general installation of Python3 found in LXPLUS etc, but needs some  additional (common) packages. Depending on your environment, the correct executable might be called either `python` or `python3` (see version with `python --version`).

Install the needed packages with

```bash
python3 -m pip install -U -r requirements.txt
```

Run analysis for the preferred example file using

```bash
python3 vdm.py -l pltlumizero /eos/cms/store/group/dpg_bril/comm_bril/vdmdata/2021/original/7525/7525_2110302352_2110310014.hd5
```

I suggest looking at the code while comparing the printouts to the code statements to get an idea what is being calculated.

Next, Look into folder `output` and compare it to the contents of `framework_results_without_beam_current_calibration`.

Also try flags `-pdf`, `-cbc`, `-fit` and other values for `-l`. See help with `python3 vdm.py --help` 

## List of features
- Choice of luminometer
- Choice of fit
- Beam current calibration
- Fit pdfs
- Multiple files as input. Give a list of files as unnamed parameter or pipe the output of `ls` with ` ls *.hd5 | xargs python3 vdm.py -l <lumi> `
