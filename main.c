#include "nbody.h"

int main(int argc, char *argv[])
{
  const double dt = init();
  const int    ns = 100; /* number of sub-steps between outputs */
  int i; /* sub-step index */

  for(;;) {
    dump();
    for(i = 0; i < ns; ++i)
      step(dt);
  }

  return 0;
}
