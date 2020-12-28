cmake_minimum_required(VERSION 3.0.0)
project(darknet VERSION 0.1.0)

set(GPU FALSE)
set(CUDNN FALSE)
set(OPENCV FALSE)
set(OPENMP FALSE)
set(DEBUG TRUE)

set(VPATH "./src/" "./examples")
set(SLIB libdarknet.so)
set(ALIB libdarknet.a)
set(EXEC darknet)
set(OBJDIR "./obj/")

set(CC gcc)
set(CPP g++)
set(NVCC nvcc)
set(AR ar)
set(ARFLAGS rcs)
set(OPTS -Ofast)
set(LDFLAGS -lm -pthread)
set(COMMON -Iinclude/ -Isrc/)
set(CFLAGS -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC)

if(OPENMP) 
 set(CFLAGS "${CFLAGS} -fopenmp")
endif()

if(DEBUG)
 set(OPTS "-O0 -g")
endif()

set(CFLAGS "${CFLAGS} ${OPTS}")

if(OPENCV)
 set(COMMON "${COMMON} -DOPENCV")
 set(CFLAGS "${CFLAGS} -DOPENCV")
 set(LDFLAGS "${LDFLAGS} `pkg-config --libs opencv` -lstdc++")
 set(COMMON "${COMMON} `pkg-config --cflags opencv`")
endif()

if(GPU)
 set(COMMON "{COMMON} -DGPU -I/usr/local/cuda/include/")
 set(CFLAGS "{CFLAGS} -DGPU")
 set(LDFLAGS "{LDFLAGS} -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand")
endif()

if(CUDNN)
 set(COMMON "{COMMON} -DCUDNN")
 set(CFLAGS "{CFLAGS} -DCUDNN")
 set(LDFLAGS "{LDFLAGS} -lcudnn")
endif()

add_executable(darknet main.cpp)

