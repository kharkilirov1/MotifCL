pybind11_add_module(_motifcl bindings.cpp)
target_link_libraries(_motifcl PRIVATE motifcl)
set_target_properties(_motifcl PROPERTIES
    OUTPUT_NAME "_motifcl"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/motifcl"
)
motifcl_apply_compiler_options(_motifcl)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/motifcl DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/kernels DESTINATION ${CMAKE_BINARY_DIR}/motifcl)
add_custom_target(motifcl_python_copy_package ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/motifcl"
        "${CMAKE_BINARY_DIR}/motifcl"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/kernels"
        "${CMAKE_BINARY_DIR}/motifcl/kernels"
    COMMENT "Copying MotifCL Python package and kernels to the build tree"
)
add_dependencies(_motifcl motifcl_python_copy_package)

install(TARGETS _motifcl
    LIBRARY DESTINATION motifcl
    RUNTIME DESTINATION motifcl
    ARCHIVE DESTINATION motifcl
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/motifcl/ DESTINATION motifcl FILES_MATCHING PATTERN "*.py")
install(DIRECTORY ${PROJECT_SOURCE_DIR}/kernels DESTINATION motifcl)
