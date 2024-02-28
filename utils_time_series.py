from datetime import timedelta, date, datetime

import pandas as pd


def create_index_map(t_max, start_date='2018-01-01', end_date=None, freq='W-Mon'):
    frequencies = {'D', 'W-Mon'}

    if freq not in frequencies:
        raise ValueError(f"{freq} is not allowed")
    if end_date:
        raise NotImplementedError('Not implemented yet')

    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()

    # First create end_date
    if freq == 'W-Mon':
        end_date = pd.to_datetime(start_date + timedelta(days=t_max * 7))
    elif freq == 'D':
        end_date = pd.to_datetime(start_date + timedelta(days=t_max))
    else:
        return

    indices = dict()
    indices['index'] = pd.RangeIndex(t_max + 1)
    indices['timestamp'] = pd.date_range(start=start_date, end=end_date, freq=freq)

    index_map = {
        ('index', 'timestamp'): dict(zip(indices['index'], indices['timestamp'])),
        ('timestamp', 'index'): dict(zip(indices['timestamp'], indices['index']))
    }

    return index_map, indices
