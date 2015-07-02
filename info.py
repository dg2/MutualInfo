'''
Information measures

author: Dario Garcia (dario.garcia@cba.com.au)
'''
import pandas as pd
import numpy as np
import sys
import preproc


def entropy_discrete(x):
    p = pd.value_counts(x) / float(len(x))
    ent = -1 * np.nansum(p * np.log(p))
    return ent


def MI_discrete(feature, target, normalise=True):
    target = target.dropna()
    counts = feature.ix[target.index].value_counts()
    aux = target.groupby(feature).apply(entropy_discrete)
    ent = entropy_discrete(target)
    MI = ent - ((counts * aux) / sum(counts)).sum()
    if normalise is True:
        MI = MI / ent
    return MI


def MI_analysis(df, target, min_prop=0.05, max_bins=10, max_target_values=5, bin_kind='tree'):
    '''Discrete mutual information analysis between predictors
    and target.

    If a column does not look discrete (i.e. the number of
    values is over max_bins), then it is discretized into
    max_bins buckets (if numerical) or value with less than
    min_prop relative frequency are grouped together (if string).

    Datetime columns are automatically expanded.

    Parameters
    ----------
    df: DataFrame

    target: Series or str

    min_prop: float, default 0.05

    max_bins: int, default 10

    max_target_values: int, default 5

    bin_kind: 'quantiles', 'fixed_width', 'tree', default 'tree'
    '''
    if isinstance(target, str):
        target = df[target]

    if len(target.unique()) > max_target_values:
        if (target.dtype == 'O') or (np.issubdtype(target.dtype, np.datetime64)):
            target = preproc.bundle_categories(target, max_categories=max_target_values)
            sys.stderr.write('The target variable has too many possible values. Limiting the analysis to the top %i\n' % max_target_values)
        else:
            target = preproc.to_bins(target, max_target_values)

    target = target.dropna()

    out = {}

    df = df.ix[target.index]
    counts = df.count()

    for col in df.columns:
        if df[col].dtype == 'O':
            if len(df[col].unique()) > max_bins:
                aux = preproc.bundle_categories(df[col], min_prop=min_prop)
                if len(aux.dropna().unique()) < 2:
                    sys.stderr.write('Discarding column %s\n' % col)
                    continue
                out[col] = MI_discrete(aux, target)
            else:
                out[col] = MI_discrete(df[col], target)
        elif np.issubdtype(df[col].dtype, np.datetime64):
            # Working with dates
            aux = preproc.explode_dates(df[col])
            for aux_col in aux.columns:
                out[aux_col] = MI_discrete(aux[aux_col], target)
        else:
            # Should be numeric
            if len(df[col].unique()) > max_bins:
                aux = preproc.to_bins(df[col], max_bins, kind=bin_kind, target=target)
                out[col] = MI_discrete(aux, target)
            else:
                out[col] = MI_discrete(df[col], target)

    out = pd.Series(out).fillna(0)
    out = pd.DataFrame(out, columns=['MI'])
    if len(out) > 0:
        out['counts'] = counts
    return out.sort('MI', ascending=False)


def contrast(df, target_column):
    gr = df.groupby(target_column)
    print gr

    def info(x):
        pass
