'''
Preprocessing utilities

author: Dario Garcia (dario.garcia@cba.com.au)
'''
import pandas as pd
import numpy as np
import dateutil
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree


def bundle_categories(series, min_count=None, min_prop=None, counts=None, bundle_value='other', inplace=False,
                      ignore_na=True, max_categories=None):
    '''Bundle together all of the values of a feature whose frequencies are below a certain threshold.

    Returns a Series object with the new values.

    Parameters
    ----------

    series: Series
        Original feature

    min_count: int, default None
        Minimum number of occurences of a value for it to be
        preserved

    min_prop: float, default None
        Minimum frequency (in [0,1]) of a value for it to be
        preserved. Frequency is calculated based on the total
        number of samples or only on non-nan entries depending
        on the ignore_na parameter

    counts: map or Series, default None
        Precomputed counts for the different feature values.
        If None, it will be calculated internally

    bundle_value: any, default 'other'
        Value to substitute for all those whose frequency falls
        below the minimum

    inplace: Boolean, default False
        Modify in place or create a new copy

    ignore_na: Boolean, default True
        Ignore NA values when obtaining relative frequencies

    max_categories: int, default None
    '''
    if (min_count is None) and (min_prop is None) and (max_categories is None):
        raise ValueError('Must specify min_count, min_prop or max_categories')

    if not (min_prop is None):
        if ignore_na is True:
            l = series.count()
        else:
            l = len(series)

        min_count = float(l) * min_prop

    if counts is None:
        counts = series.value_counts()

    if not (max_categories is None):
        bundled_values = counts.index[max_categories - 1:]
    elif isinstance(counts, pd.Series):
        bundled_values = counts.index[counts < min_count]
    else:
        bundled_values = [k for k in counts.keys() if counts[k] < min_count]

    if inplace is False:
        series = series.copy()

    series[series.isin(bundled_values)] = bundle_value

    return series


def sparsity_filter(df, min_count=None, min_prop=None, counts=None, inplace=False):
    ''' Filter out the columns in a DataFrame whose sparsity level
    is above a threshold.

    Parameters
    ----------
    df: DataFrame
        Original data

    min_count: int, default None
        Minimum number of non-NA values in a columns for it
        to be preserved

    min_prop: float, default None
        Minimum proportion (in [0,1]) of non-NA values for
        a column to be preserved.

    counts: map or Series, default None
        Precalculated counts

    inplace: Boolean, default False
    '''
    if (min_count is None) and (min_prop is None):
        raise ValueError('Must specify min_count or min_prop')

    if min_count is None:
        min_count = len(df) * float(min_prop)

    if counts is None:
        counts = df.count()

    if isinstance(counts, pd.Series):
        filtered_cols = counts[counts < min_count].index.values
    else:
        filtered_cols = [k for k in counts.keys() if counts[k] < min_count]

    if inplace is True:
        df.drop(filtered_cols, axis=1, inplace=True)
    else:
        df = df.drop(filtered_cols, axis=1)

    return df


def is_date(series):
    '''
    Heuristically determines if a column contains dates
    '''
    pass


def explode_datetime(series, hour=False, day_of_month=False):
    '''
    Explode a series of dates into a DataFrame containing separate columns for
    year, month, day, day of week

    Parameters
    ----------
    series: Series

    hour: Boolean, default False

    day_of_month: Boolean, default False

    Return
    ------
    Dataframe with the required columns. The column names
    are prefixed by the input series name
    '''

    aux = lambda x: [x.year, x.month, x.day, x.dayofweek, x.hour]
    columns = ['year', 'month', 'day', 'day_of_week', 'hour']

    columns = map(lambda x: series.name + '_' + x, columns)
    tmp = pd.DataFrame(series.map(aux).tolist(), columns=columns)

    if hour is False:
        tmp.drop(series.name + '_' + 'hour', axis=1, inplace=True)
    if day_of_month is False:
        tmp.drop(series.name + '_' + 'day', axis=1, inplace=True)
    return tmp


def to_datetime(series, fix_century=True, max_year=datetime.now().year):
    '''Converts a series of strings representing dates to date objects.

    This function is a thin wrapper around dateutil.parser.parse which
    makes it less prone to overparsing and more robust to ambiguous cases.

    Returns a series of datetime64[ns] objects

    Parameters
    ----------
    series: Series
        A series of strings to try to parse into datetime objects

    fix_century: Boolean, default True
        The dateutil parser usually interpret future years when working
        with two-digit representations. We fix that by going back one
        century when we encounter years larger than max_year

    max_year: Int, default datetime.now().year
    '''
    NaT = pd.tslib.NaT

    def dummy_replace(**x):
        if len(x) == 0:
            return NaT
        else:
            return datetime(1970, 1, 1).replace(**x)

    NaT.replace = dummy_replace

    out = series.map(lambda x: dateutil.parser.parse(x, default=NaT))

    if fix_century is True:
        idx = out > datetime(max_year, 1, 1)
        out[idx] = out[idx].map(lambda x: x.replace(year=x.year - 100))

    return out


def expand_dummies(df, cols, dummy_na=False):
    if isinstance(cols, str):
        cols = [cols]
    new_dfs = []
    for col in cols:
        new_dfs.append(pd.get_dummies(df[col], prefix=col, prefix_sep='=', dummy_na=dummy_na))

    return pd.concat(new_dfs, axis=1)


def to_bins(series, num_bins, kind='quantiles', value_range=1, inplace=False, target=None, remove_inf=True):
    '''
    Parameters
    ----------
    series: Series

    num_bins: int

    kind: 'quantiles', 'fixed_with', 'tree'

    target: Series, default None
        Target for tree-based quantization

    value_range: float, default 1

    inplace: Boolean, default False

    remove_inf: Boolean, default True
    '''
    # Parameter checking
    if (kind == 'tree') and (target is None):
        raise ValueError('A target must be specified for tree-based binning')
    if (value_range < 0) or (value_range > 1):
        raise ValueError('The value range parameter must be in [0,1]')

    if value_range == 1:
        min_val = series.min()
        max_val = series.max()
    else:
        min_val = series.quantile(1 - value_range)
        max_val = series.quantile(value_range)

    if inplace is False:
        series = series.copy()

    if remove_inf is True:
        series = series.replace([np.inf, -np.inf], np.nan)

    series = series.dropna()

    v = series.values

    if kind == 'tree':
        # TODO: Use a classifier or regressor depending on whether the target
        # looks categorical or continuous
        aux = pd.get_dummies(target.dropna())
        for i, col in enumerate(aux.columns):
            aux[col][aux[col] == 1] = i
        labels = aux.sum(axis=1)
        labels = labels[series.index].dropna()
        clf = DecisionTreeClassifier(max_depth=np.ceil(np.log(num_bins)))
        clf.fit(series.ix[labels.index].values[:, np.newaxis], labels.values)
        out = clf.tree_.apply(series.values[:, np.newaxis].astype(_tree.DTYPE))
        out = pd.Series(data=out, index=series.index)
    elif kind == 'fixed_width':
        out = (num_bins - 1) * np.ones(len(series), dtype=int)
        bins = np.zeros(num_bins + 1)
        bins[1:-1] = np.linspace(min_val, max_val, num_bins - 1)

        for i in range(num_bins):
            try:
                out[(v <= bins[i + 1]) & (v > bins[i])] = i
            except:
                pass
    elif kind == 'quantiles':
        out = pd.Series(data=num_bins - 1, index=series.index, name=series.name + '_bins')
        ss = series.order()
        samples_per_bin = out.count() / num_bins
        for i in range(num_bins):
            out.ix[ss.index[(samples_per_bin * i):(samples_per_bin * (1 + i))]] = i
    else:
        raise ValueError('Unknown binning method: %s' % kind)

    return out


def analyse_feature(series, continuous_thresh=0.25, categorical_component_thresh=0.05):
    N = series.count()
    counts = series.value_counts()
    D = len(counts)  # Number of distinct values
    if (D >= N * continuous_thresh):
        print 'Looks continuous'
        # Check for categorical components
        print counts[counts > N * categorical_component_thresh]
    else:
        print 'Looks categorical'
