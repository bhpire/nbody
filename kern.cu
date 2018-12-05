#include <math.h>
#include "nbody.h"

extern Z n;

V *dev_r, *dev_v;

static __global__ void kick(V *v, V *r, R dt, Z n)
{
  __shared__ V cache[TILE];
  V dt_a = {0.0, 0.0, 0.0};
  V self = {0.0, 0.0, 0.0};

  Z i = blockIdx.x * blockDim.x + threadIdx.x; /* my index */
  Z j;

  /* If my index is valid, load my position */
  if(i < n) self = r[i];

  /* For each block ... */
  for(j = 0; j < gridDim.x; ++j) {
    Z I = j * blockDim.x + threadIdx.x; /* the index of the other particle */
    Z k;

    /* If the index of the other particle is valid, load its position
       into the cache.  Otherwise, use my position.  Note that
       self-interaction vanishes because of softening */
    cache[threadIdx.x] = (I < n) ? r[I] : self;
    __syncthreads();

    /* A subset of the particles are loaded into the cache.  We can
       now compute my interaction with them. */
    #pragma unroll 8
    for(k = 0; k < TILE; ++k) {
      V other  = cache[k];

      V dr     = {self.x - other.x, self.y - other.y, self.z - other.z};
      R rr     = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + SOFTENING2;
      R dt_rrr = dt / (rr * sqrt(rr));

      dt_a.x -= dr.x * dt_rrr;
      dt_a.y -= dr.y * dt_rrr;
      dt_a.z -= dr.z * dt_rrr;
    }
  }

  if(i < n) {
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
  const int block_sz = TILE;
  const int grid_sz = (n + TILE - 1) / TILE;

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
