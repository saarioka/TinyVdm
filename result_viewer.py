"""
Call this with 'python result_viewer.py' to launch an interactive plotting tool
See help with python 'result_viewer.py --help'
"""
import argparse
from pathlib import Path
from glob import glob

import pandas as pd
from pandasgui import show

def main(args) -> None:
    if Path(args.filename).is_file():
        data = pd.read_csv(args.filename)
        print(f'Read {args.filename}')
    else:
        files = glob('output/results/*.csv')
        data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).reset_index(drop=True)
        if data.empty:
            print('No files found - make sure to run analysis first!')
            exit(1)
        else:
            print('Combined result file missing, loaded individual files')
    show(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot result files. Default is to show all scans, but you can give a filename manually with flag -f/--file')
    parser.add_argument('-f', '--filename', type=str, help='Csv file to be analysed, something like "output/results/result_?.csv"', default='output/results.csv')
    main(parser.parse_args())
