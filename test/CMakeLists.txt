cmake_minimum_required(VERSION 3.10)

include(CTest)

include(${PROJECT_SOURCE_DIR}/cmake/DownloadProject/DownloadProject.cmake)
download_project(PROJ googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG master
                UPDATE_DISCONNECTED 1
)

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR} ${googletest_SOURCE_DIR})

add_executable(TestAll test.cpp)
target_link_libraries(TestAll gtest_main)
add_test(NAME TestAll COMMAND TestAll)
