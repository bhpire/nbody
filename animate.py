#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm

if len(sys.argv) == 1:
    print("usage: {} [input]".format(sys.argv[0]))
    exit(0)

input = sys.argv[1]
r = np.genfromtxt(input)

fig, ax = plt.subplots()
plot,   = ax.plot(r[:10,0], r[:10,1])
ax.set_aspect('equal')
ax.set_xlim([-1.2,1.2])
ax.set_ylim([-1.2,1.2])

def update(i):
    plot.set_xdata(r[i:i+10,0])
    plot.set_ydata(r[i:i+10,1])
    return plot

ani = anm.FuncAnimation(fig, update)

plt.show()
