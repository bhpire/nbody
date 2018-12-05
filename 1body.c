#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[])
{
  const double dt = 0.001; /* time step size */
  const int ns = 100; /* number of sub-steps between outputs */
  int i; /* sub-step index */

  double rx = 1.0, ry = 0.0, rz = 0.0; /* initial position */
  double vx = 0.0, vy = 1.0, vz = 0.0; /* initial velocity */

  /* Keep looping until output error or aborted, i.e., Ctrl-C */
  for(;;) {
    printf("%g %g %g\n", rx, ry, rz);

    for(i = 0; i < ns; ++i) {
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
  }

  return 0;
}
