import logging

import numpy as np


def from_h5(f, table, what, condition=None, mask=None):
    """Wrapper for data retrieval from H5 files"""
    if mask is None:
        return np.array([row[what] for row in f.root[table]] if condition is None else [row[what] for row in f.root[table].where(condition)])
    else:
        return np.array([row[what][mask] for row in f.root[table]] if condition is None else [row[what][mask] for row in f.root[table].where(condition)])


def get_nice_name_for_luminometer(luminometer) -> str:
    if luminometer.startswith('scan'):
        luminometer = luminometer[6:]

    rate_table_to_lumi = {
        'pltlumizero': 'PLT',
        'bcm1flumi': 'BCM1F',
        'hfetlumi': 'HFET',
        'hfoclumi': 'HFOC',
        }

    return rate_table_to_lumi[luminometer] if luminometer in rate_table_to_lumi.keys() else luminometer.upper()


def init_logger(level: int) -> None:
    if level == 5:
        # These modules give a bit too verbose DEBUG output (they support logging module)
        for package in ('matplotlib', 'numba'):
            logging.getLogger(package).setLevel(logging.INFO)

    level_str_to_int = {
        1: logging.CRITICAL,
        2: logging.ERROR,
        3: logging.WARNING,
        4: logging.INFO,
        5: logging.DEBUG,
    }

    logging.basicConfig(format='%(filename)s l.%(lineno)d - %(levelname)s: %(message)s', level=level_str_to_int[level])

