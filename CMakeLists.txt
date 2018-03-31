cmake_minimum_required(VERSION 3.10)

project(sfm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NOTE: -fsanitize=addressをつけるとEigenが落ちるので注意
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Wsign-compare -fsanitize=undefined -fno-common -fwrapv")

option(ENABLE_TESTING "Enable testing" ON)
option(LIBCXX_DEBUG "Enable debug mode of standard c++ library" ON)

include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen3_3_4)

if(LIBCXX_DEBUG)
add_definitions(-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC)
endif(LIBCXX_DEBUG)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif(ENABLE_TESTING)
