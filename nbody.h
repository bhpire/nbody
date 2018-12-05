#ifndef NBODY_H
#define NBODY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

#define TILE (64)

typedef struct {
  double x, y, z;
} vector;

typedef int Z;
#if defined(DOUBLE) || defined(OUBLE) /* so -DOUBLE works */
typedef double R;
#define SOFTENING2 (1.0e-6)
#define SQRT(x) sqrt(x)
#else
typedef float R;
#define SOFTENING2 (1.0e-6f)
#define SQRT(x) sqrtf(x)
#endif
typedef struct {
  R x, y, z;
} V;

double init(int);
void evol(int, double);
int dump(FILE *);

#ifdef __cplusplus
}
#endif

#endif
