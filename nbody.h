#ifndef NBODY_H
#define NBODY_H

#include <stdio.h>

double init(void);
void evol(int, double);
int dump(FILE *);

#endif
