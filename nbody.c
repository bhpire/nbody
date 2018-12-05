#include <math.h>
#include <stdlib.h>
#include "nbody.h"

static int n;
static vector *r, *v;

static vector sphere(void)
{
  vector r;
  double x, y, z, rr;

  do {
    x  = 2.0 * rand() / RAND_MAX - 1.0;
    y  = 2.0 * rand() / RAND_MAX - 1.0;
    z  = 2.0 * rand() / RAND_MAX - 1.0;
    rr = x * x + y * y + z * z;
  } while(rr > 1.0);

  r.x = x;
  r.y = y;
  r.z = z;
  return r;
}

double init(int N)
{
  vector p = {0.0, 0.0, 0.0};
  double cbrt_n;
  int i;

  n      = N;
  cbrt_n = pow(n, 1.0/3.0);

  r = (vector *)malloc(sizeof(vector) * n);
  v = (vector *)malloc(sizeof(vector) * n);

  /* Random initial condition */
  for(i = 0; i < n; ++i) {
    r[i] = sphere();
    v[i] = sphere();

    p.x += v[i].x *= cbrt_n;
    p.y += v[i].y *= cbrt_n;
    p.z += v[i].z *= cbrt_n;
  }

  /* Make sure the net momentum is zero */
  p.x /= n;
  p.y /= n;
  p.z /= n;

  for(i = 0; i < n; ++i) {
    v[i].x -= p.x;
    v[i].y -= p.y;
    v[i].z -= p.z;
  }

  return 1.0e-5 / cbrt_n; /* time step size */
}

int dump(FILE *file)
{
  int i;
  for(i = 0; i < n - 1; ++i)
    fprintf(file, "%g %g %g, ", r[i].x, r[i].y, r[i].z);
  fprintf(file, "%g %g %g\n", r[i].x, r[i].y, r[i].z);

  return 1;
}

static inline void kick(double dt)
{
  int i, j;

  for(i = 0; i < n; ++i)
    for(j = i + 1; j < n; ++j) {
      vector dr     = {r[i].x - r[j].x, r[i].y - r[j].y, r[i].z - r[j].z};
      double rr     = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + SOFTENING2;
      double dt_rrr = dt / (rr * sqrt(rr));

      v[i].x -= dr.x * dt_rrr;
      v[i].y -= dr.y * dt_rrr;
      v[i].z -= dr.z * dt_rrr;

      v[j].x += dr.x * dt_rrr;
      v[j].y += dr.y * dt_rrr;
      v[j].z += dr.z * dt_rrr;
    }
}

static inline void drift(double dt)
{
  int i;

  for(i = 0; i < n; ++i) {
    r[i].x += v[i].x * dt;
    r[i].y += v[i].y * dt;
    r[i].z += v[i].z * dt;
  }
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
