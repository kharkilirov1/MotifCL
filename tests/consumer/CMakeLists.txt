cmake_minimum_required(VERSION 3.20)
project(motifcl_consumer_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MotifCL CONFIG REQUIRED)

add_executable(motifcl_consumer_smoke main.cpp)
target_link_libraries(motifcl_consumer_smoke PRIVATE MotifCL::motifcl)
