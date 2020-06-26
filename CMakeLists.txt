cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_COMPILER g++-9)


project(fitting)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenMP
find_package(OpenMP REQUIRED)

# Eigen
set(EIGEN_INCLUDE_DIRS "~/libraries/include/eigen_pr")

# Open3D
list(APPEND CMAKE_INSTALL_PREFIX "~/libraries/")
find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
list(APPEND Open3D_LIBRARIES dl)

# OpenSubdiv
set(OSD_INCLUDE_DIRS "/usr/local/include")
set(OSD_LIBRARY_DIRS "/usr/local/lib")
set(OSD_LIBS "${OSD_LIBRARY_DIRS}/libosdCPU.a")

# json
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

include_directories(${EIGEN_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS} ${OSD_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS} ${OSD_LIBRARY_DIRS} ${JSONCPP_LIBRARY_DIRS})

add_executable(
    fit_sphere
    sphere_fitting/fit_sphere.cpp
    fitting/fitting_functor.cpp
    subdiv/evaluator.cpp
    mesh/mesh.cpp
)
target_link_libraries(fit_sphere ${Open3D_LIBRARIES} ${OSD_LIBS} ${JSONCPP_LIBRARIES} OpenMP::OpenMP_CXX c++)


add_executable(
    fit_hand
    hand_fitting/fit_hand.cpp
    hand_fitting/mano.cpp
    fitting/fitting_functor.cpp
    subdiv/evaluator.cpp
    mesh/mesh.cpp
)
target_link_libraries(fit_hand ${Open3D_LIBRARIES} ${OSD_LIBS} ${JSONCPP_LIBRARIES} OpenMP::OpenMP_CXX c++)

add_executable(
    hand_expt
    expts/generate_hand_results.cpp
    hand_fitting/mano.cpp
    fitting/fitting_functor.cpp
    subdiv/evaluator.cpp
    mesh/mesh.cpp
)
target_link_libraries(hand_expt ${Open3D_LIBRARIES} ${OSD_LIBS} ${JSONCPP_LIBRARIES} OpenMP::OpenMP_CXX c++)

add_executable(
    sphere_expt
    expts/generate_sphere_results.cpp
    fitting/fitting_functor.cpp
    subdiv/evaluator.cpp
    mesh/mesh.cpp
)
target_link_libraries(sphere_expt ${Open3D_LIBRARIES} ${OSD_LIBS} ${JSONCPP_LIBRARIES} OpenMP::OpenMP_CXX c++)
