compile:
	gcc main.c 1body.c -lm -O0 -Wall -Wextra -o 1body0
	gcc main.c 1body.c -lm -O3 -Wall -Wextra -o 1body3

clean:
	-rm -f 1body?
	-rm -f *.out
