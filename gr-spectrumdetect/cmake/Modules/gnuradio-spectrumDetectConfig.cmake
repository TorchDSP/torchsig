find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_SPECTRUMDETECT gnuradio-spectrumDetect)

FIND_PATH(
    GR_SPECTRUMDETECT_INCLUDE_DIRS
    NAMES gnuradio/spectrumDetect/api.h
    HINTS $ENV{SPECTRUMDETECT_DIR}/include
        ${PC_SPECTRUMDETECT_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_SPECTRUMDETECT_LIBRARIES
    NAMES gnuradio-spectrumDetect
    HINTS $ENV{SPECTRUMDETECT_DIR}/lib
        ${PC_SPECTRUMDETECT_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-spectrumDetectTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_SPECTRUMDETECT DEFAULT_MSG GR_SPECTRUMDETECT_LIBRARIES GR_SPECTRUMDETECT_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_SPECTRUMDETECT_LIBRARIES GR_SPECTRUMDETECT_INCLUDE_DIRS)
