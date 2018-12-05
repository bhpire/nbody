#include <math.h>
#include "nbody.h"

static vector r1, r2;
static vector v1, v2;

double init(void)
{
  r1.x = 1.0; r1.y = 0.0; r1.z = 0.0;
  v1.x = 0.0; v1.y = 0.6; v1.z = 0.0;

  r2.x =-1.0; r2.y = 0.0; r2.z = 0.0;
  v2.x = 0.0; v2.y =-0.6; v2.z = 0.0;

  return 1.0e-4; /* time step size */
}

int dump(FILE *file)
{
  return fprintf(file, "%g %g %g, %g %g %g\n",
                 r1.x, r1.y, r1.z, r2.x, r2.y, r2.z) > 0;
}

void evol(int n, double dt)
{
  vector dr;
  double rr, kdt_rrr;

  double kdt = dt / 2; /* the first kick is a half step */
  int i;

  for(i = 0; i < n; ++i) {
    /* Get force and kick */
    dr.x = r1.x - r2.x;
    dr.y = r1.y - r2.y;
    dr.z = r1.z - r2.z;

    rr = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    kdt_rrr = kdt / (rr * sqrt(rr));

    v1.x -= dr.x * kdt_rrr;
    v1.y -= dr.y * kdt_rrr;
    v1.z -= dr.z * kdt_rrr;

    v2.x += dr.x * kdt_rrr;
    v2.y += dr.y * kdt_rrr;
    v2.z += dr.z * kdt_rrr;

    /* Drift */
    r1.x += v1.x * dt;
    r1.y += v1.y * dt;
    r1.z += v1.z * dt;

    r2.x += v2.x * dt;
    r2.y += v2.y * dt;
    r2.z += v2.z * dt;

    /* all other kicks are full steps */
    kdt = dt;
  }
  /* Last half-step correction */
  kdt = dt / 2;

  /* Get force and kick */
  dr.x = r1.x - r2.x;
  dr.y = r1.y - r2.y;
  dr.z = r1.z - r2.z;

  rr = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
  kdt_rrr = kdt / (rr * sqrt(rr));

  v1.x += dr.x * kdt_rrr;
  v1.y += dr.y * kdt_rrr;
  v1.z += dr.z * kdt_rrr;

  v2.x -= dr.x * kdt_rrr;
  v2.y -= dr.y * kdt_rrr;
  v2.z -= dr.z * kdt_rrr;
}
