import logging


def get_nice_name_for_luminometer(luminometer):
    rate_table_to_lumi = {
        'pltlumizero': 'PLT',
        'bcm1flumi': 'BCM1F',
        'hfetlumi': 'HFET',
        'hoctlumi': 'HFOC',
        }
    if luminometer.startswith('scan'):
        luminometer = luminometer[6:]

    return rate_table_to_lumi[luminometer] if luminometer in rate_table_to_lumi.keys() else luminometer.upper()


def init_logger(level: int):
    level_str_to_int = {
        1: logging.CRITICAL,
        2: logging.ERROR,
        3: logging.WARNING,
        4: logging.INFO,
        5: logging.DEBUG,
    }

    logging.basicConfig(format='%(filename)s l.%(lineno)d - %(levelname)s: %(message)s', level=level_str_to_int[level])

