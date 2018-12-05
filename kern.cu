#include <math.h>
#include "nbody.h"

extern Z n;
extern V *r, *v;

V *dev_r, *dev_v;

static inline void kick(double dt)
{
  Z i, j;

  for(i = 0; i < n; ++i) {
    V dt_a = {0.0, 0.0, 0.0};

    for(j = 0; j < n; ++j) if(i != j) {
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

static inline void drift(R dt)
{
  Z i;

  for(i = 0; i < n; ++i) {
    r[i].x += v[i].x * dt;
    r[i].y += v[i].y * dt;
    r[i].z += v[i].z * dt;
  }
}

void evol(int n, double dt)
{
  R kdt = dt / 2; /* the first kick is a half step */
  Z i;
  for(i = 0; i < n; ++i) {
    kick(kdt);
    drift(dt);
    kdt = dt; /* all other kicks are full steps */
  }
  /* Last half-step correction */
  kick(dt / 2);
}
