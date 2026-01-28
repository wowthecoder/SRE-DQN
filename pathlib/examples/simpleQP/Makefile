# -*- Makefile -*-

.PHONY: all test

PATHDIR = ../..
PATHLIB = $(PATHDIR)/lib

CC     = gcc
CFLAGS = -I$(PATHDIR)/include -m64
SHARED_PATHLIB = -L$(PATHLIB) -lpath50
# we list the .a explicitly: the OSX linker will prefer the .dylib over the .a otherwise
STATIC_PATHLIB = $(PATHLIB)/libpath50.a

all: ijvalDriver_shared ijvalDriver_static xpDriver_shared xpDriver_static

ijvalDriver_shared: ijvalDriver.c simpleQP.c
	$(CC) $(CFLAGS) -o $@ $^ $(SHARED_PATHLIB) -lm

ijvalDriver_static: ijvalDriver.c simpleQP.c
	$(CC) $(CFLAGS) -o $@ $^ $(STATIC_PATHLIB) -lm

xpDriver_shared: xpDriver.c sparseQP.c
	$(CC) $(CFLAGS) -o $@ $^ $(SHARED_PATHLIB) -lm

xpDriver_static: xpDriver.c sparseQP.c
	$(CC) $(CFLAGS) -o $@ $^ $(STATIC_PATHLIB) -lm

test:
	@echo "-------------------------------------------------------------"
	@echo "---------- file da.dat should have obj of 21.0771 -----------"
	@echo "-------------------------------------------------------------"
	./ijvalDriver_shared sampleData/da.dat
	./ijvalDriver_static sampleData/da.dat
	@echo "-------------------------------------------------------------"
	@echo "------------- file da2.dat should be infeasible -------------"
	@echo "-------------------------------------------------------------"
	./ijvalDriver_shared sampleData/da2.dat
	./ijvalDriver_static sampleData/da2.dat
	@echo "-------------------------------------------------------------"
	@echo "-------------------------------------------------------------"
	( cd sampleData ; ../xpDriver_static tinyAscii.info )
