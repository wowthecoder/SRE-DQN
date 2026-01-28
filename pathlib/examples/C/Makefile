# -*- Makefile -*-

.PHONY: all test

PATHDIR = ../..
PATHLIB = $(PATHDIR)/lib

CC     = gcc
CFLAGS = -I$(PATHDIR)/include -m64
SHARED_PATHLIB = -L$(PATHLIB) -lpath50
# we list the .a explicitly: the OSX linker will prefer the .dylib over the .a otherwise
STATIC_PATHLIB = $(PATHLIB)/libpath50.a

all: obstacle_shared obstacle_static

obstacle_shared: obstacle.c Standalone_Path.c
	$(CC) $(CFLAGS) -o $@ $^ $(SHARED_PATHLIB) -lm

obstacle_static: obstacle.c Standalone_Path.c
	$(CC) $(CFLAGS) -o $@ $^ $(STATIC_PATHLIB) -lm

# On Ferris laptop need
# gcc -I../../include -m64 -o obstacle obstacle.c Standalone_Path.c -L../../lib -L/usr/local/lib/x86_64 -lpath50 -lgfortran -lm

test:
	rm -f path.opt
	./obstacle_shared 6
	./obstacle_static 6
	@echo "-------------------------------------------------------------"
	@echo "---- demo-size test passed: trying larger instances next ----"
	@echo "-------------------------------------------------------------"
	./obstacle_shared 20
	./obstacle_static 20
	@echo "-------------------------------------------------------------"
	@echo "----------------- testing LUSOL BLU library------------------"
	@echo "-------------------------------------------------------------"
	echo > path.opt
	echo "factorization_method blu_lusol" >> path.opt
	./obstacle_shared 20
	rm -f path.opt
ifeq ($(TESTUMFPACK),yes)
	@echo "-------------------------------------------------------------"
	@echo "------------------ testing UMFPACK library ------------------"
	@echo "-------------------------------------------------------------"
	echo > path.opt
	echo "factorization_method umfpack" >> path.opt
ifneq ($(UMFPACK_LIBNAME),/umfpack_libname/not/set)
	echo "factorization_library_name $(UMFPACK_LIBNAME)" >> path.opt
endif
	./obstacle_shared 20
	rm -f path.opt
endif
