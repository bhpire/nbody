#include <math.h>
#include <stdlib.h>
#include "nbody.h"

extern V *dev_r, *dev_v;

Z n;

static V *r, *v;

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

  cudaMalloc((void **)&dev_r, sizeof(V) * n);
  cudaMalloc((void **)&dev_v, sizeof(V) * n);
  cudaMemcpy(dev_r, r, sizeof(V) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_v, v, sizeof(V) * n, cudaMemcpyHostToDevice);

  return 1.0e-5 / cbrt_n; /* time step size */
}

int dump(FILE *file)
{
  static Z first = 1;
  Z i;

  if(first)
    first = 0;
  else
    cudaMemcpy(r, dev_r, sizeof(V) * n, cudaMemcpyDeviceToHost);

  for(i = 0; i < n - 1; ++i)
    fprintf(file, "%g %g %g, ", r[i].x, r[i].y, r[i].z);
  fprintf(file, "%g %g %g\n", r[i].x, r[i].y, r[i].z);

  return 1;
}
