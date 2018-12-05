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
name, _ = os.path.splitext(input)
r = np.genfromtxt(name+'.out' if _ == '' else input)

fig, ax = plt.subplots()
plot,   = ax.plot(r[:10,0], r[:10,1])
ax.set_aspect('equal')
ax.set_xlim([-3,3])
ax.set_ylim([-2,2])

def update(i):
    plot.set_xdata(r[i:i+10,0])
    plot.set_ydata(r[i:i+10,1])
    return plot

ani = anm.FuncAnimation(fig, update, interval=40)

ani.save(name+'.mp4')
