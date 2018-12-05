#include <math.h>
#include "nbody.h"

extern Z n;

V *dev_r, *dev_v;

static __global__ void kick(V *v, V *r, R dt, Z n)
{
  Z i = blockIdx.x * blockDim.x + threadIdx.x;
  Z j;

  if(i < n) {
    V dt_a = {0.0, 0.0, 0.0};

    for(j = 0; j < n; ++j) {
      V dr     = {r[i].x - r[j].x, r[i].y - r[j].y, r[i].z - r[j].z};
      R rr     = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + SOFTENING2;
      R dt_rrr = dt / (rr * sqrt(rr));

      dt_a.x -= dr.x * dt_rrr;
      dt_a.y -= dr.y * dt_rrr;
      dt_a.z -= dr.z * dt_rrr;
    }

    v[i].x += dt_a.x;
    v[i].y += dt_a.y;
    v[i].z += dt_a.z;
  }
}

static __global__ void drift(V *r, V *v, R dt, Z n)
{
  Z i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    r[i].x += v[i].x * dt;
    r[i].y += v[i].y * dt;
    r[i].z += v[i].z * dt;
  }
}

void evol(int ns, double dt)
{
  const int block_sz = 256;
  const int grid_sz = (n + block_sz - 1) / block_sz;

  R kdt = dt / 2; /* the first kick is a half step */
  Z i;
  for(i = 0; i < ns; ++i) {
    kick <<<grid_sz, block_sz>>>(dev_v, dev_r, kdt, n);
    drift<<<grid_sz, block_sz>>>(dev_r, dev_v,  dt, n);
    kdt = dt; /* all other kicks are full steps */
  }
  /* Last half-step correction */
  kick<<<grid_sz, block_sz>>>(dev_v, dev_r, dt / 2, n);
}
