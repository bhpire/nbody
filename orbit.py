#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    print("usage: {} [input]".format(sys.argv[0]))
    exit(0)

r = np.genfromtxt(sys.argv[1])

plt.plot(r[:,0], r[:,1])
plt.axes().set_aspect('equal')
plt.show()
