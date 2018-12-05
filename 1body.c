#include <stdio.h>
#include <math.h>
#include "nbody.h"

static double rx, ry, rz;
static double vx, vy, vz;

double init(void)
{
  rx = 1.0, ry = 0.0, rz = 0.0; /* initial position */
  vx = 0.0, vy = 1.0, vz = 0.0; /* initial velocity */

  return 1.0e-3; /* time step size */
}

int dump(void)
{
  return printf("%g %g %g\n", rx, ry, rz) > 0;
}

void evol(int n, double dt)
{
  int i;
  for(i = 0; i < n; ++i) {
    double rr, rrr, ax, ay, az;

    /* Get force */
    rr = rx * rx + ry * ry + rz * rz;
    rrr= rr * sqrt(rr);
    ax = - rx / rrr;
    ay = - ry / rrr;
    az = - rz / rrr;

    /* Kick */
    vx += ax * dt / 2;
    vy += ay * dt / 2;
    vz += az * dt / 2;

    /* Drift */
    rx += vx * dt;
    ry += vy * dt;
    rz += vz * dt;

    /* Get new force */
    rr = rx * rx + ry * ry + rz * rz;
    rrr= rr * sqrt(rr);
    ax = - rx / rrr;
    ay = - ry / rrr;
    az = - rz / rrr;

    /* Kick again */
    vx += ax * dt / 2;
    vy += ay * dt / 2;
    vz += az * dt / 2;
  }
}
