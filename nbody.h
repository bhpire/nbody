#ifndef NBODY_H
#define NBODY_H

#include <stdio.h>

#define SOFTENING2 (1.0e-6)

typedef struct {
  double x, y, z;
} vector;

double init(int);
void evol(int, double);
int dump(FILE *);

#endif
