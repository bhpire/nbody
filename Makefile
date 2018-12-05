compile:
	gcc main.c 1body.c -lm -O0 -Wall -Wextra -o 1body

clean:
	-rm -f 1body
