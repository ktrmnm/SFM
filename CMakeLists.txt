cmake_minimum_required(VERSION 3.10)

project(sfm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-v")

option(ENABLE_TESTING  "Enable testing" ON)

include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen3_3_4)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif(ENABLE_TESTING)
