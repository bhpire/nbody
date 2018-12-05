#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    print("usage: {} [input]".format(sys.argv[0]))
    exit(0)

input = sys.argv[1]
name, _ = os.path.splitext(input)
r = np.genfromtxt(name+'.out' if _ == '' else input)
n = r.shape[1] // 3

for i in range(n):
    plt.plot(r[:,i*3+0], r[:,i*3+1])
plt.axes().set_aspect('equal')

plt.savefig(name+'.png')
