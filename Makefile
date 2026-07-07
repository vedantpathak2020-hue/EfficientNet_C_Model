CC = gcc
CFLAGS = -Iinclude -O3 -Wall -march=native -mtune=native -flto -funroll-loops
SRC = src/main.c src/operators.c
OBJ = $(SRC:.c=.o)

all: efficientnet_inference

efficientnet_inference: $(OBJ)
	$(CC) $(OBJ) -o efficientnet_inference -lm
	
# This rule compiles your .c files into .o (object) files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	./efficientnet_inference

clean:
	rm -f src/*.o efficientnet_inference