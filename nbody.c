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
  double cbrt_n;
  int i;

  n      = N;
  cbrt_n = pow(n, 1.0/3.0);

  r = (vector *)malloc(sizeof(vector) * n);
  v = (vector *)malloc(sizeof(vector) * n);

  for(i = 0; i < n; ++i) {
    r[i] = sphere();
    v[i] = sphere();

    v[i].x *= cbrt_n;
    v[i].y *= cbrt_n;
    v[i].z *= cbrt_n;
  }

  return 1.0e-4 / cbrt_n; /* time step size */
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

  for(i = 0; i < n; ++i) {
    vector dt_a = {0.0, 0.0, 0.0};

    for(j = 0; j < n; ++j) if(i != j) {
      vector dr     = {r[i].x - r[j].x, r[i].y - r[j].y, r[i].z - r[j].z};
      double rr     = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
      double dt_rrr = dt / (rr * sqrt(rr));

      dt_a.x -= dr.x * dt_rrr;
      dt_a.y -= dr.y * dt_rrr;
      dt_a.z -= dr.z * dt_rrr;
    }

    v[i].x += dt_a.x;
    v[i].y += dt_a.y;
    v[i].z += dt_a.z;
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
