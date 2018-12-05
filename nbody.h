#ifndef NBODY_H
#define NBODY_H

#include <stdio.h>

typedef struct {
  double x, y, z;
} vector;

double init(int);
void evol(int, double);
int dump(FILE *);

#endif
