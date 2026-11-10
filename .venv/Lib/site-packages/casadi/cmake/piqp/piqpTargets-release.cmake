#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "piqp::piqp_c" for configuration "Release"
set_property(TARGET piqp::piqp_c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(piqp::piqp_c PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libpiqpc.dll.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS piqp::piqp_c )
list(APPEND _IMPORT_CHECK_FILES_FOR_piqp::piqp_c "${_IMPORT_PREFIX}/lib/libpiqpc.dll.a" )

# Import target "piqp::piqp" for configuration "Release"
set_property(TARGET piqp::piqp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(piqp::piqp PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libpiqp.dll.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS piqp::piqp )
list(APPEND _IMPORT_CHECK_FILES_FOR_piqp::piqp "${_IMPORT_PREFIX}/lib/libpiqp.dll.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
