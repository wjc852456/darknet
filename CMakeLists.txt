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

include_directories("include" "src")

aux_source_directory("src" SRC_FILES)

set(OBJ gemm.c utils.c cuda.c deconvolutional_layer.c convolutional_layer.c list.c image.c activations.c im2col.c col2im.c blas.c crop_layer.c dropout_layer.c maxpool_layer.c softmax_layer.c data.c matrix.c network.c connected_layer.c cost_layer.c parser.c option_list.c detection_layer.c route_layer.c upsample_layer.c box.c normalization_layer.c avgpool_layer.c layer.c local_layer.c shortcut_layer.c logistic_layer.c activation_layer.c rnn_layer.c gru_layer.c crnn_layer.c demo.c batchnorm_layer.c region_layer.c reorg_layer.c tree.c  lstm_layer.c l2norm_layer.c yolo_layer.c iseg_layer.c image_opencv.cpp)

set(EXECOBJA captcha.c lsd.c super.c art.c tag.c cifar.c go.c rnn.c segmenter.c regressor.c classifier.c coco.c yolo.c detector.c nightmare.c instance-segmenter.c darknet.c)

file(GLOB EXAMPLE_SRCS examples/EXECOBJA)
file(GLOB SRC_SRCS src/OBJ)

foreach(f ${OBJ})
 set(OBJ_SRCS ${OBJ_SRCS} src/${f})
endforeach(f)

foreach(f ${EXECOBJA})
 set(EXECOBJA_SRCS ${EXECOBJA_SRCS} examples/${f})
endforeach(f)

add_library(ALIB STATIC ${OBJ_SRCS})
#add_library(${SLIB} SHARED ${OBJ_SRCS})
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ALIB PUBLIC Threads::Threads)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
link_libraries(ALIB m)
add_executable(cmake_darknet ${EXECOBJA_SRCS})


