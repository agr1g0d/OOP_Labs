cmake_minimum_required(VERSION 3.16)

project(OOPLabs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB LAB_DIRECTORIES CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/Lab*"
)

set(ALL_LAB_SOURCES)
set(ALL_HEADER_DIRECTORIES)
foreach(LAB_DIRECTORY IN LISTS LAB_DIRECTORIES)
    set(LAB_SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${LAB_DIRECTORY}/src")

    if(IS_DIRECTORY "${LAB_SOURCE_DIRECTORY}")
        file(GLOB LAB_SOURCES CONFIGURE_DEPENDS
            "${LAB_SOURCE_DIRECTORY}/*.cpp"
        )
        list(APPEND ALL_LAB_SOURCES ${LAB_SOURCES})
    endif()

    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${LAB_DIRECTORY}/headers")
        list(APPEND ALL_HEADER_DIRECTORIES
            "${CMAKE_CURRENT_SOURCE_DIR}/${LAB_DIRECTORY}/headers"
        )
    endif()
endforeach()

if(ALL_LAB_SOURCES)
    add_executable(OOPLabs ${ALL_LAB_SOURCES})
    target_include_directories(OOPLabs PRIVATE
        ${ALL_HEADER_DIRECTORIES}
    )
endif()

enable_testing()

add_executable(dataset_tests
    tests/dataset_tests.cpp
    Lab1/src/dataset.cpp
    Lab1/src/matrix.cpp
)
target_include_directories(dataset_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Lab1/headers"
    "${CMAKE_CURRENT_SOURCE_DIR}/Lab2/headers"
)

add_test(
    NAME dataset_tests
    COMMAND dataset_tests "${CMAKE_CURRENT_SOURCE_DIR}/data/Autism-Adult-Data.csv"
)
