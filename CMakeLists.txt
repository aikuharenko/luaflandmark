CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()
FIND_PACKAGE(Torch REQUIRED)

#opencv
find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIR} "${PROJECT_SOURCE_DIR}/libflandmark")

set( CMAKE_CXX_FLAGS "-fpermissive" )

SET(src flandmark.cpp)
SET(luasrc init.lua flandmark_model.dat haarcascade_frontalface_alt.xml)

ADD_TORCH_PACKAGE(flandmark "${src}" "${luasrc}" "flandmark")
TARGET_LINK_LIBRARIES(flandmark flandmark_shared ${OpenCV_LIBS})
