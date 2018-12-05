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

void step(double dt)
{
  double ax, ay, az;

  /* Get force */
  ax = - rx / pow(rx * rx + ry * ry + rz * rz, 1.5);
  ay = - ry / pow(rx * rx + ry * ry + rz * rz, 1.5);
  az = - rz / pow(rx * rx + ry * ry + rz * rz, 1.5);

  /* Kick */
  vx += ax * dt / 2;
  vy += ay * dt / 2;
  vz += az * dt / 2;

  /* Drift */
  rx += vx * dt;
  ry += vy * dt;
  rz += vz * dt;

  /* Get new force */
  ax = - rx / pow(rx * rx + ry * ry + rz * rz, 1.5);
  ay = - ry / pow(rx * rx + ry * ry + rz * rz, 1.5);
  az = - rz / pow(rx * rx + ry * ry + rz * rz, 1.5);

  /* Kick again */
  vx += ax * dt / 2;
  vy += ay * dt / 2;
  vz += az * dt / 2;
}
