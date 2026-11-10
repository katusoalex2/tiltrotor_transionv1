#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "antlr4_shared" for configuration "Release"
set_property(TARGET antlr4_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(antlr4_shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libantlr4-runtime.dll.a"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libantlr4-runtime.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS antlr4_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_antlr4_shared "${_IMPORT_PREFIX}/lib/libantlr4-runtime.dll.a" "${_IMPORT_PREFIX}/bin/libantlr4-runtime.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
