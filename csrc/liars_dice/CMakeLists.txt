cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cfr)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -I .. -Wall -mcmodel=medium -march=native -Wextra -Wno-register -fPIC -Wfatal-errors -O3")
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")

# Build GTest.
# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_SOURCE_DIR}/../../third_party/googletest third_party/googletest EXCLUDE_FROM_ALL)

# find packages & third_party
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

#### TORCH LIB  #####
if (NOT TARGET torch_library)
  execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
  )
  list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
  # set cuda arch list so that the built binary can be run on both pascal and volta
  set(TORCH_CUDA_ARCH_LIST "6.0;7.0")
  find_package(Torch REQUIRED)
endif()

add_library(liars_dice_lib liars_dice subgame_solving real_net recursive_solving stats)
target_link_libraries(liars_dice_lib torch)
set_target_properties(liars_dice_lib PROPERTIES CXX_STANDARD 17)

add_executable(recursive_eval recursive_eval)
target_link_libraries(recursive_eval liars_dice_lib)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pybind11 third_party/pybind11
)

add_library(_rela
  rela/types.cc
)
target_include_directories(_rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(_rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(_rela PUBLIC $ENV{CONDA_PREFIX}/include)
target_link_libraries(_rela torch)

# python lib
pybind11_add_module(rela rela/pybind.cc)
target_link_libraries(rela PUBLIC _rela liars_dice_lib)

add_executable(gen_benchmark gen_benchmark)
target_link_libraries(gen_benchmark liars_dice_lib _rela)

#################
# Tests
include(GoogleTest)
enable_testing()

add_executable(liar_game_test liars_dice_test.cc)
target_link_libraries(liar_game_test liars_dice_lib gtest_main)
add_test(NAME liar_game COMMAND liar_game_test)

add_executable(liar_tree_test tree_test.cc)
target_link_libraries(liar_tree_test liars_dice_lib gtest_main)
add_test(NAME liar_tree COMMAND liar_tree_test)

add_executable(liar_recursive_solving_test recursive_solving_test.cc)
target_link_libraries(liar_recursive_solving_test liars_dice_lib gtest_main)
add_test(NAME liar_recursive_solving COMMAND liar_recursive_solving_test)

add_executable(liar_subgame_solving_test subgame_solving_test.cc)
target_link_libraries(liar_subgame_solving_test liars_dice_lib gtest_main)
add_test(NAME liar_subgame_solving COMMAND liar_subgame_solving_test)
