cmake_minimum_required(VERSION 3.6)
project(kingfisher)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    apufunktiot.c
    apufunktiot.h
    bitmatrice.c
    bitmatrice.h
    brules.c
    brules.h
    constr.c
    constr.h
    error.c
    error.h
    glob.c
    glob.h
    ibitvector.h
    kingfisher.c
        measures.c
    measures.h
    trie.c
    trie.h)

add_executable(kingfisher ${SOURCE_FILES})