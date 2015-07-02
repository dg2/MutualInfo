#!/usr/bin/env python

import sys

import pandas as pd
import numpy as np

import info

SEP = '\t'

current_feat = None
data_block = []
MAX_BINS = 8
    
def process_block(data_block):
    # Put the block in a DataFrame
    df = pd.DataFrame(data_block, columns=['feat_id', 'label', 'value'])
    print '%s\t%f' % (df['feat_id'][0], df['value'].sum())


for line in sys.stdin:
    line = line.strip()
    # Collect data for a given feature
    fields = line.split(SEP) 
    if len(fields) != 3:
        continue
    
    feat_id = fields[0]
    label = int(fields[1])
    try:
        feat_value = float(fields[2])
    except:
#        sys.stderr.write("Can't parse %s (feat. %s)" % (fields[2], fields[0]))
        feat_value = np.nan

    if (feat_id != current_feat):
        if not (current_feat is None):
            process_block(data_block)

        
        data_block = []
        current_feat = feat_id

    data_block.append((feat_id, label, feat_value))

if len(data_block) > 0:
    process_block(data_block)
