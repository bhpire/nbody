#include <stdio.h>
#include <math.h>
#include "nbody.h"

static double rx, ry, rz;
static double vx, vy, vz;

double init(void)
{
  rx = 1.0, ry = 0.0, rz = 0.0; /* initial position */
  vx = 0.0, vy = 1.2, vz = 0.0; /* initial velocity */

  return 1.0e-4; /* time step size */
}

int dump(void)
{
  return printf("%g %g %g\n", rx, ry, rz) > 0;
}

void evol(int n, double dt)
{
  double kdt = dt / 2; /* the first kick is a half step */
  int i;
  double rr, kdt_rrr;
  for(i = 0; i < n; ++i) {
    /* Get force and kick */
    rr = rx * rx + ry * ry + rz * rz;
    kdt_rrr = kdt / (rr * sqrt(rr));
    vx -= rx * kdt_rrr;
    vy -= ry * kdt_rrr;
    vz -= rz * kdt_rrr;

    /* Drift */
    rx += vx * dt;
    ry += vy * dt;
    rz += vz * dt;

    /* all other kicks are full steps */
    kdt = dt;
  }
  /* Last half-step correction */
  kdt = dt / 2;

  /* Get force and kick */
  rr = rx * rx + ry * ry + rz * rz;
  kdt_rrr = kdt / (rr * sqrt(rr));
  vx -= rx * kdt_rrr;
  vy -= ry * kdt_rrr;
  vz -= rz * kdt_rrr;
}
