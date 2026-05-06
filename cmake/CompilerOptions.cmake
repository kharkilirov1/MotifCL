function(motifcl_apply_compiler_options target_name)
    target_compile_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
        $<$<BOOL:${MOTIFCL_WARNINGS_AS_ERRORS}>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<BOOL:${MOTIFCL_WARNINGS_AS_ERRORS}>:$<$<CXX_COMPILER_ID:MSVC>:/WX>>
    )
    if(MOTIFCL_ENABLE_SANITIZERS AND NOT MSVC)
        target_compile_options(${target_name} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target_name} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()
