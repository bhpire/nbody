#include <math.h>
#include <stdlib.h>
#include "nbody.h"

Z n;
V *r, *v;

static V sphere(void)
{
  V r;
  R x, y, z, rr;

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
  V p = {0.0, 0.0, 0.0};
  R cbrt_n;
  Z i;

  n      = N;
  cbrt_n = pow(n, 1.0/3.0);

  r = (V *)malloc(sizeof(V) * n);
  v = (V *)malloc(sizeof(V) * n);

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
  Z i;
  for(i = 0; i < n - 1; ++i)
    fprintf(file, "%g %g %g, ", r[i].x, r[i].y, r[i].z);
  fprintf(file, "%g %g %g\n", r[i].x, r[i].y, r[i].z);

  return 1;
}
