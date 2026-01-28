# -*- Makefile -*-

.PHONY: all test

PATHDIR = ../..
PATHLIB = $(PATHDIR)/lib

CXX    = g++
CXXFLAGS = -I$(PATHDIR)/include -m64
SHARED_PATHLIB = -L$(PATHLIB) -lpath50
# we list the .a explicitly: the OSX linker will prefer the .dylib over the .a otherwise
STATIC_PATHLIB = $(PATHLIB)/libpath50.a

all: obstacle_c_shared obstacle_c_static

obstacle_c_shared: obstacle_c.cpp cWrapper_Path.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(SHARED_PATHLIB) -lm

obstacle_c_static: obstacle_c.cpp cWrapper_Path.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(STATIC_PATHLIB) -lm

test:
	./obstacle_c_shared 6
	./obstacle_c_static 6
	@echo "-------------------------------------------------------------"
	@echo "---- demo-size test passed: trying larger instances next ----"
	@echo "-------------------------------------------------------------"
	./obstacle_c_shared 20
	./obstacle_c_static 20
