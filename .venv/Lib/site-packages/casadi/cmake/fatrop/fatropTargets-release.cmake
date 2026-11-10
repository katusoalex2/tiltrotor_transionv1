#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "fatrop::fatrop" for configuration "Release"
set_property(TARGET fatrop::fatrop APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(fatrop::fatrop PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libfatrop.dll.a"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libfatrop.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS fatrop::fatrop )
list(APPEND _IMPORT_CHECK_FILES_FOR_fatrop::fatrop "${_IMPORT_PREFIX}/lib/libfatrop.dll.a" "${_IMPORT_PREFIX}/bin/libfatrop.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
