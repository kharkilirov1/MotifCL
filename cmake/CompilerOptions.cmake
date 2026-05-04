function(motifcl_apply_compiler_options target_name)
    target_compile_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
        $<$<BOOL:${MOTIFCL_WARNINGS_AS_ERRORS}>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<BOOL:${MOTIFCL_WARNINGS_AS_ERRORS}>:$<$<CXX_COMPILER_ID:MSVC>:/WX>>
    )
endfunction()
