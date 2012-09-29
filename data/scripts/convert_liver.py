"""
This script is used to change the labels 1, 2 to -1, 1 respectively
in the liver dataset.

Usage:
$ python convert_liver.py original converted

Danushka Bollegala
2012/05/22.
"""

import sys

fin = open(sys.argv[1], 'r')
fout = open(sys.argv[2], 'w')

for line in fin:
    p = line.strip().split()
    label = int(p[0])
    if label == 1:
        convLabel = -1
    elif label == 2:
        convLabel = 1
    else:
        raise ValueError
    fout.write("%s %s\n" % (str(convLabel), " ".join(p[1:])))

fin.close()
fout.close()
