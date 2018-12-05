compile:
	gcc main.c 1body.c -lm -O0 -Wall -Wextra -o 1body0
	gcc main.c 1body.c -lm -O3 -Wall -Wextra -o 1body3
	gcc main.c 2body.c -lm -O0 -Wall -Wextra -o 2body0
	gcc main.c 2body.c -lm -O3 -Wall -Wextra -o 2body3
	gcc main.c 3body.c -lm -O0 -Wall -Wextra -o 3body0
	gcc main.c 3body.c -lm -O3 -Wall -Wextra -o 3body3
	gcc main.c nbody.c -lm -O0 -Wall -Wextra -o nbody0
	gcc main.c nbody.c -lm -O3 -Wall -Wextra -o nbody3

clean:
	-rm -f ?body?
	-rm -f *.out
