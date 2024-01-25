cmake_minimum_required(VERSION 3.13)

# Project name and description
project(
    schrodint VERSION 0.1
    DESCRIPTION "A library to perform multi-precision numerical integration of ODEs."
    LANGUAGES CXX
    )

# Release build type by default
if ( NOT CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Compile flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# C++ standard options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#include_directories(include)
add_executable(test tests/test.cpp)

#target_link_libraries(
#    exponential PUBLIC 
#    cln
#    ginac 
#    )

# Boost
#set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREADED ON)
#set(Boost_USE_STATIC_RUNTIME OFF)

#find_package(Boost 1.69 
#    COMPONENTS program_options
#    REQUIRED
#    )
#
#if ( Boost_FOUND ) 
#    include_directories(${Boost_INCLUDE_DIRS})
#endif()


target_link_libraries(
    test PUBLIC 
    #    Boost::boost
    #    Boost::program_options
    cln
    ginac 
    )
