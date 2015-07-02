#!/usr/bin/env python
import sys

SEP = '\t'
LABEL_COL = 2

for line in sys.stdin:
    feats = line.strip().split(SEP)
    label = feats[LABEL_COL]
    for i, feat in enumerate(feats):
        print '%i\t%s\t%s' % (i, label, feat)

