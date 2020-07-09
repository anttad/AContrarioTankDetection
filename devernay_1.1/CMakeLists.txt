cmake_minimum_required(VERSION 2.8)



# SETUP A MAKE CLEAN TO CLEAN EVERYTHING
#
#
set_directory_properties (PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES 
      "CMakeCache.txt;CMakeFiles;Makefile;cmake_install.cmake")


if (NOT TARGET IIOLIB)
   include_directories("iio")
   add_subdirectory(iio)
endif() 
SET( LIBS ${LIBS} IIOLIB)



SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ")


#
include_directories("mw3")
add_executable (devernay devernay_cmd.c devernay.c io.c)
target_link_libraries(devernay ${LIBS} )


#SET_TARGET_PROPERTIES(stereoSAD PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp") 

