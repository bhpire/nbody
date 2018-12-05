#include <math.h>
#include "nbody.h"

static vector r1, r2, r3;
static vector v1, v2, v3;

double init(void)
{
  r1.x = 1.0; r1.y = 0.0; r1.z = 0.0;
  v1.x = 0.0; v1.y = 1.0; v1.z = 0.0;

  r2.x =-1.0; r2.y = 0.0; r2.z = 0.0;
  v2.x = 0.0; v2.y =-1.0; v2.z = 0.0;

  r3.x = 0.0; r3.y = 0.0; r3.z = 0.0;
  v3.x = 0.0; v3.y = 0.0; v3.z = 0.0;

  return 1.0e-4; /* time step size */
}

int dump(FILE *file)
{
  return fprintf(file, "%g %g %g, %g %g %g, %g %g %g\n",
                 r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z) > 0;
}

static inline void kick(double dt)
{
  vector dt_a12, dt_a23, dt_a31;

  {
    vector dr = {r1.x - r2.x, r1.y - r2.y, r1.z - r2.z};
    double rr = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    double dt_rrr = dt / (rr * sqrt(rr));
    dt_a12.x = dr.x * dt_rrr;
    dt_a12.y = dr.y * dt_rrr;
    dt_a12.z = dr.z * dt_rrr;
  }

  {
    vector dr = {r2.x - r3.x, r2.y - r3.y, r2.z - r3.z};
    double rr = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    double dt_rrr = dt / (rr * sqrt(rr));
    dt_a23.x = dr.x * dt_rrr;
    dt_a23.y = dr.y * dt_rrr;
    dt_a23.z = dr.z * dt_rrr;
  }

  {
    vector dr = {r3.x - r1.x, r3.y - r1.y, r3.z - r1.z};
    double rr = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    double dt_rrr = dt / (rr * sqrt(rr));
    dt_a31.x = dr.x * dt_rrr;
    dt_a31.y = dr.y * dt_rrr;
    dt_a31.z = dr.z * dt_rrr;
  }

  v1.x += dt_a31.x - dt_a12.x;
  v1.y += dt_a31.y - dt_a12.y;
  v1.z += dt_a31.z - dt_a12.z;
  v2.x += dt_a12.x - dt_a23.x;
  v2.y += dt_a12.y - dt_a23.y;
  v2.z += dt_a12.z - dt_a23.z;
  v3.x += dt_a23.x - dt_a31.x;
  v3.y += dt_a23.y - dt_a31.y;
  v3.z += dt_a23.z - dt_a31.z;
}

static inline void drift(double dt)
{
  r1.x += v1.x * dt;
  r1.y += v1.y * dt;
  r1.z += v1.z * dt;

  r2.x += v2.x * dt;
  r2.y += v2.y * dt;
  r2.z += v2.z * dt;

  r3.x += v3.x * dt;
  r3.y += v3.y * dt;
  r3.z += v3.z * dt;
}

void evol(int n, double dt)
{
  double kdt = dt / 2; /* the first kick is a half step */
  int i;
  for(i = 0; i < n; ++i) {
    kick(kdt);
    drift(dt);
    kdt = dt; /* all other kicks are full steps */
  }
  /* Last half-step correction */
  kick(dt / 2);
}
