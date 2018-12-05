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
n = r.shape[1] // 3

fig, ax = plt.subplots()
lines = []
for i in range(n):
    l, = ax.plot(r[:10,i*3+0], r[:10,i*3+1])
    lines.append(l)
ax.set_aspect('equal')
ax.set_xlim([-3,3])
ax.set_ylim([-2,2])

def update(j):
    for i, l in enumerate(lines):
        l.set_data(r[j:j+10,i*3+0], r[j:j+10,i*3+1])
    return lines

ani = anm.FuncAnimation(fig, update, interval=40)

ani.save(name+'.mp4')
