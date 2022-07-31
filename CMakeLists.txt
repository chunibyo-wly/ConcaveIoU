project(cuda_gtest_plugin_examples)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})

set( CUDA_ARCH "sm_86" CACHE STRING "CUDA architecture to compile for" )
set( GTEST_HOME $ENV{GTEST_HOME} CACHE FILEPATH "Path to gtest root directory (assumes libgtest.a is placed in $GTEST_HOME/build)" )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories( ${GTEST_HOME}/googletest/include )
link_directories( ${GTEST_HOME}/build/lib )

find_package(PythonInterp REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=${CUDA_ARCH} 
                        -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ 
                        -D_GLIBCXX_USE_CXX11_ABI=1 -lpthread" )
 
cuda_add_executable(example_cuda_gtest main.cc src/example_test.cu)
target_link_libraries(example_cuda_gtest pthread)
target_link_libraries(example_cuda_gtest libgtest.a ${TORCH_LIBRARIES})

cuda_add_executable(convex_gtest main.cc src/convex_test.cu)
target_link_libraries(convex_gtest pthread)
target_link_libraries(convex_gtest libgtest.a ${TORCH_LIBRARIES})

cuda_add_executable(concave_gtest main.cc src/concave_test.cu)
target_link_libraries(concave_gtest pthread)
target_link_libraries(concave_gtest libgtest.a ${TORCH_LIBRARIES})
