#include "nbody.h"
#include <stdio.h>
#include <signal.h>
#include <sys/time.h>

static volatile int running = 1;

void handler(int signal)
{
  running = 0;
  fputs("STOP", stderr);
}

int main(int argc, char *argv[])
{
  const double dt = init();
  const int    ns = 100; /* number of sub-steps between outputs */

  struct timeval t0, t1;
  double wtime; /* "wall" clock time between each outputs */

  signal(SIGINT, handler);

  while(running) {
    dump();

    gettimeofday(&t0, NULL);
    evol(ns, dt);
    gettimeofday(&t1, NULL);

    wtime = (t1.tv_sec - t0.tv_sec) + 1.0e-6 * (t1.tv_usec - t0.tv_usec);
    fprintf(stderr, "%g ns/step\n", 1.0e9 * wtime / ns);
  }

  return 0;
}
