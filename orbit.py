#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    print("usage: {} [input]".format(sys.argv[0]))
    exit(0)

input = sys.argv[1]
r = np.genfromtxt(input)

plt.plot(r[:,0], r[:,1])
plt.axes().set_aspect('equal')

name, _ = os.path.splitext(input)
plt.savefig(name+'.png')
