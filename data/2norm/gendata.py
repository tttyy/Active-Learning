#! /usr/bin/env python

import sys

inf = open(sys.argv[1], 'r')
of = open(sys.argv[2], 'w')
c = int(sys.argv[3])

lines = inf.readlines()
for line in lines:
    if(c > 0):
        lab=1
    else:
        lab=-1
    c -= 1
    of.write('%s,%d\n' % (line[:-1], lab))

of.close()
inf.close()
