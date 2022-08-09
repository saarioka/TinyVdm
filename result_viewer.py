"""
Call this with 'python result_viewer.py' to launch an interactive plotting tool
"""
from pathlib import Path
import pandas as pd
from pandasgui import show
from glob import glob

def main():
    result_filename = 'output/result.csv'
    if Path(result_filename).is_file():
        data = pd.read_csv(result_filename)
        print(f'Read {result_filename}')
    else:
        files = glob('output/results/*.csv')
        data = [pd.read_csv(f) for f in files]
        data = pd.concat(data, ignore_index=True).reset_index(drop=True)
        if data.empty:
            print('No file found - make sure to run analysis first!')
            exit(1)
        else:
            print('Combined result file missing, loaded individual files')
    show(data)

if __name__ == '__main__':
    main()

