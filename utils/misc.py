import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from dateutil.relativedelta import *

import fnmatch
import os

import json
import re

from collections.abc import Iterable

def get_start_end_dates(dt, _freq):
    '''
    Returns a pair of two lists; start and end. A consecutive period starts from start[n] and ends on end[n].
    
    Parameters:
    dt : Series
        Its values are dates and type is Timestamp 
    
    _freq : string
        either 'M' for monthly or 'W' for weekly
        
    month_end : Boolean
        Change dates in "end" list to %Y-%d-LastDay from %Y-%d-01.
        
    Returns:
    The first list contains start dates and the second list contains end dates.

    Examples:
    ---------
    rec_dt = pd.Series(rec_months[rec_months==1].index.to_timestamp(), name='rec_date')

    >> rec_dt
        rec_date
        1926-10-30/1926-11-05   1926-11-05 23:59:59.999999999
        1926-11-06/1926-11-12   1926-11-12 23:59:59.999999999
    
    rec_starts, rec_ends = get_start_end_dates(rec_dt[rec_dt>='1956'], _freq)

    '''
    
    if dt.empty:
        return [], []
    
    n = 0
    start = []
    end = []
    start.append(dt.iloc[0])
    prev = dt.iloc[0]
    
    unit_period = relativedelta(months=1) if _freq == 'M' else relativedelta(weeks=1)
    done_flag = False
    
    for d in dt.iloc[1:]:
        if done_flag:
            # As done_flag is marked, we add a start date into the `start` list.
            start.append(prev)
            done_flag = False

        if d != (prev + unit_period):
            # if dates are not consecutive, than it means it's an end point.
            if _freq == 'M':
                end.append(prev + MonthEnd(0))
            elif _freq == 'W':
                end.append(prev)
            
            # Mark this flag as true so that we can append a start date.
            done_flag = True
        
        prev = d
        
    end.append(d)

    return start, end


def get_filenames(fname, path='.'):
    '''
    Returns a list of file names in the path `path`

    Parameters:
    fname : str
        either a file name or filename-pattern that contains Unix shell-style wildcards.
        e.g. 'news*.json' will return news(1).json, news(2).json, news_.json and so forth.
    
    path : string
        path where this function looks for at. Default is a current path where this function is being called.
    
    Returns:
    A list of file names that match a pattern specified in `fname`
    '''
    filenames = []
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, fname):
            filenames.append(file)

    return filenames

def get_df_from_json_chunks(filenames, path='.'):
    '''
    Returns a DataFrame instance where its contents are extracted from json files specified in `filenames`.

    Parameters:
    filenames : list
        A list of file names. You may use get_filenames() to get it.
    
    path: str
        path where this function looks for at. Default is a current path where this function is being called.
    '''
    df = pd.DataFrame()

    for filename in filenames:
        with open(path+filename, 'r', encoding='UTF-8') as f:
            raw_json_chunks = f.readlines()
            assert len(raw_json_chunks) == 1, filename + 'does not consist of a single line'
            json_chunks = re.findall('\[\{(.*?)\}\]', raw_json_chunks[0])
            records = ['[{' +  json_chunk + '}]' for json_chunk in json_chunks]
            print('{:s}: {:d} json chunk(s), each of which has...'.format(filename, len(json_chunks)))
            for no, record in enumerate(records):
                new_df = pd.read_json(record)
                print('\t#{:d}: {:d} record(s).'.format(no+1, new_df.shape[0]))
                if ~df.empty:
                    df = pd.concat([df, new_df])
                else:
                    df = new_df.copy()

    df = df.reset_index(drop=True)
    print('\nTotal: {:d} records'.format(df.shape[0]))

    return df

def get_excel_column_name(i, letter='A'):
    """Returns the alphabet that is `i`-th apart from `letter`, which is compatible with the column name in Excel.
    e.g.: get_letter(26) returns 'AA
    
    Parameters:
    -----------
    letter : character
        The letter from which `i`-th apart.
    
    i : int
        `i`-th apart from `letter.
    """
    if i <= 25:
        return chr(ord(letter)+i)
    else:
        return get_excel_column_name((i // 26)-1) + get_excel_column_name(i % 26)


def get_last_bday(date_from=None, holidays=[''], weekmask='Mon Tue Wed Thu Fri'):
    """Returns the last business day from `date_from`.

    Parameters:
    -----------
    date_from : string ('%Y-%m-%d')

    holidays : list
        list/array of dates to exclude from the set of valid business days,
        passed to ``numpy.busdaycalendar``

    weekmask : str, Default 'Mon Tue Wed Thu Fri'
        weekmask of valid business days, passed to ``numpy.busdaycalendar``

    Returns:
    --------
    string
        The last business day
    """
    
    
    bday = pd.tseries.offsets.CustomBusinessDay(holidays=holidays, weekmask=weekmask)
    if date_from is None:
        date_from = pd.Timestamp.today()

    return (date_from - bday).strftime('%Y-%m-%d')


def get_fred_asof(df, col_nm, asof_date, freq='M'):
    '''
    Returns data in `df` available at a point of time specified by `as_of_date`.
    
    Parameters:
    -----------
    df : DataFrame
        an object that's returned from fred.get_series_all_releases()
    
    col_nm: string
        A column name.
    
    asof_date: string or datetime
        a point of time
    
    freq: string
        'M' for a monthly frequency (default)
        'W' for a weekly frequency

    
    Returns:
    --------
    DataFrame
        its index is reference date and column name is `col_nm`
    
    Example:
    --------
    ip_all = fred.get_series_all_releases('INDPRO')
    ip_idx = get_fred_asof(ip_all, 'IP_idx', '2020-05-31')

    '''

    # We extract conditions for 'realtime_start', 'date' so that we can extract data available at a point of time.
    cond_inc_rev = df.loc[df.realtime_start <= asof_date, ['realtime_start', 'date']].groupby(by='date').max().reset_index()
    
    # Filter `df` with `cond_inc_rev`
    ef = cond_inc_rev.merge(df, left_on=['date', 'realtime_start'], right_on=['date', 'realtime_start'], how='left')
    
    if freq == 'M':
        # Convert %m-01 format to %m-the last day of that month.
        ef['date'] = ef.date + MonthEnd(0)
    
    ef.set_index('date', drop=True, inplace=True)
    ef.drop('realtime_start', axis=1, inplace=True)
    
    ef.columns = [col_nm]
    
    return ef

def get_fred_asof_history(df_all, start_date, col_nm, freq='M', end_date=pd.Timestamp.today()):
    '''
    Returns a dictionary where its key is a date to extract a Fred series as of that date.

    Parameters:
    -----------
    df_all: DataFrame
        an object that's returned from fred.get_series_all_releases()
        This will be passed to get_fred_asof()

    start_date: string in %Y-%m-%d format.

    freq: string
        'M' for a monthly frequency (default)
        'W' for a weekly frequency
    
    end_date: string in %Y-%m-%d format.
    '''
    if freq == 'M':
        dt_range = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq='M').strftime('%Y-%m-%d').to_list() + [pd.Timestamp.today().strftime('%Y-%m-%d')]
    elif freq == 'W':
        dt_range = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq='W-FRI').strftime('%Y-%m-%d').to_list()

    kdf_asof = {}

    for dt in dt_range:
        key = dt[:-3] if freq == 'M' else dt
        kdf_asof[key] = get_fred_asof(df_all, col_nm, dt, freq)
        
        if freq == 'M':
            kdf_asof_index = pd.to_datetime(kdf_asof[key].index).to_period('M')
        elif freq == 'W':
            kdf_asof_index = pd.to_datetime(kdf_asof[key].index - pd.Timedelta('1 days')).to_period('W-FRI')
        
        kdf_asof[key] = kdf_asof[key].set_index(kdf_asof_index, drop=True)
    
    return kdf_asof


def move_col(df, cols_to_move=[], ref_col='', place='After'):
    '''
    Moves columns specified in `cols_to_move` (After|Before) a column referred by `ref_col`.

    Parameters:
    df: DataFrame
        a DataFrame instance to which this function is applied.
    
    cols_to_move: list
        a list of column names to be moved.
    
    ref_col: string
        a column name as a reference column.
    
    place: string
        an either value of 'After' or 'Before'. Default is 'After'
    
    Source: https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5
    '''
    
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])


def iterable(obj):
    '''
    Returns if obj is iterable.

    The isinstance() has been recommended already earlier, but the general consensus has been that using iter() would be better.
    If we'd use insinstance(), we wouldn't accidentally consider Faker instances (or any other objects having only __getitem__) to be iterable:
    Source: https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    '''
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def tokenize(s):
    '''Splits a string into tokens'''
    WHITESPACE = re.compile('\s+')
    return WHITESPACE.split(s)


def untokenize(s):
    '''Joins tokens into a string'''
    return ' '.join(s)

class SMWrapper(BaseEstimator, RegressorMixin):
    """
    A universal sklearn-style wrapper for statsmodels regressors 
    
    Source: https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
    """
    def __init__(self, model_class, lasso_t, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.lasso_t = lasso_t
        
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit_regularized(method='sqrt_lasso', refit=True, zero_tol=self.lasso_t)
        
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


class LogWrapper():
    def __init__(self, logger):
        self.logger = logger

    def info(self, *args, sep=' '):
        self.logger.info(sep.join("{}".format(a) for a in args))

    def debug(self, *args, sep=' '):
        self.logger.debug(sep.join("{}".format(a) for a in args))

    def warning(self, *args, sep=' '):
        self.logger.warning(sep.join("{}".format(a) for a in args))

    def error(self, *args, sep=' '):
        self.logger.error(sep.join("{}".format(a) for a in args))

    def critical(self, *args, sep=' '):
        self.logger.critical(sep.join("{}".format(a) for a in args))

    def exception(self, *args, sep=' '):
        self.logger.exception(sep.join("{}".format(a) for a in args))

    def log(self, *args, sep=' '):
        self.logger.log(sep.join("{}".format(a) for a in args))



class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


class StandardScalerClipper(StandardScaler):
    '''
    Applies StandardScaler() and then clips the scaled results outside an interval [`zmin`, `zmax`] to the interval edges.
    A default interval is [-3, 3]
    
    Source: http://flennerhag.com/2017-01-08-Recursive-Override/
    '''
    def __init__(self, zmin=-3, zmax=3):
        super(StandardScalerClipper, self).__init__(self)
        self.zmin=zmin
        self.zmax=zmax
    
    def transform(self, x):
        '''
        We simply call the same `transform` method first.
        Then we do clipping after that.
        '''
        z = super(StandardScalerClipper, self).transform(x)
        
        return np.clip(z, self.zmin, self.zmax)

