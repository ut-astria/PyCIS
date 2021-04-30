# useful resources:
# 	https://www.cs.swarthmore.edu/~newhall/unixhelp/howto_C_libraries.html
#   https://www.topbug.net/blog/2019/10/28/makefile-template-for-a-shared-library-in-c-with-explanations/
#   http://nuclear.mutantstargoat.com/articles/make/#a-makefile-for-99-of-your-programs

# define flags (note CFLAGS -MMD and -I$(INCLUDE))
CC = gcc
INCLUDES = ../gsl/include  
LIBRARIES = ../gsl/lib 
CFLAGS = -MMD -fPIC -Wall -Wextra -O3 -g -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-function -Wno-sign-compare -Wno-maybe-uninitialized -fopenmp -I$(INCLUDES)
LDFLAGS = -shared -fopenmp -lgsl -lgomp -lgslcblas -fopenmp
RM = rm -f
TARGET_LIB = libpycis_all.so

# list source files and generate objects and dependencies (via MMD)
SRCS = $(wildcard *.c) 
OBJS = $(SRCS:.c=.o)
DEPS = $(OBJS:.o=.d)

# define target shared library object 
# "make" is a default alias to "make all"
.PHONY: all
	all: %{TARGET_LIB}

# build target @ by applying ldflags with MMD to all objects
$(TARGET_LIB): $(OBJS)
	$(CC) -o $@ $^ ${LDFLAGS} -L$(LIBRARIES)

-include $(DEPS)

# clean up the temporary rule files and individual objects
# call this process by "make clean"
.PHONY: clean 
clean: 
	${RM} ${TARGET_LIB} ${OBJS}  ${DEPS}
