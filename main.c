#include "nbody.h"
#include <stdio.h>
#include <signal.h>

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
  int i; /* sub-step index */

  signal(SIGINT, handler);

  while(running) {
    dump();
    for(i = 0; i < ns; ++i)
      step(dt);
  }

  return 0;
}
