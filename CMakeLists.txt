cmake_minimum_required(VERSION 3.0.0)
project(darknet VERSION 0.1.0)

set(GPU FALSE)
set(CUDNN FALSE)
set(OPENCV FALSE)
set(OPENMP FALSE)
set(DEBUG TRUE)

set(VPATH "./src/" "./examples")
set(SLIB darknet)
set(ALIB darknet)
set(EXEC darknet)
set(OBJDIR "./obj/")

set(CC gcc)
set(CPP g++)
set(NVCC nvcc)
set(AR ar)
set(ARFLAGS rcs)
set(OPTS -Ofast)
set(LDFLAGS "-lm -pthread")
set(COMMON -Iinclude/ -Isrc/)
set(CMAKE_C_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC")

if(OPENMP) 
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
endif()

if(DEBUG)
 set(OPTS "-O0 -g")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPTS}")

if(OPENCV)
 set(COMMON "${COMMON} -DOPENCV")
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DOPENCV")
 set(LDFLAGS "${LDFLAGS} `pkg-config --libs opencv` -lstdc++")
 set(COMMON "${COMMON} `pkg-config --cflags opencv`")
endif()

if(GPU)
 set(COMMON "{COMMON} -DGPU -I/usr/local/cuda/include/")
 set(CMAKE_C_FLAGS "{CMAKE_C_FLAGS} -DGPU")
 set(LDFLAGS "{LDFLAGS} -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand")
endif()

if(CUDNN)
 set(COMMON "{COMMON} -DCUDNN")
 set(CMAKE_C_FLAGS "{CMAKE_C_FLAGS} -DCUDNN")
 set(LDFLAGS "{LDFLAGS} -lcudnn")
endif()

if(GPU)
 set(LDFLAGS "{LDFLAGS} -lstdc++")
endif()

include_directories("include")
include_directories("src")

aux_source_directory("src" SRC_FILES)

add_library(ALIB STATIC ${SRC_FILES})
add_library(SLIB SHARED ${SRC_FILES})

file(GLOB TEST_SRCS examples/*.c)
add_executable(darknet ${TEST_SRCS})

