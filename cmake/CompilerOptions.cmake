function(motifcl_apply_compiler_options target_name)
    target_compile_definitions(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
    )
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

    if(MOTIFCL_ENABLE_FAST_MATH)
        target_compile_options(${target_name} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffast-math>
            $<$<CXX_COMPILER_ID:MSVC>:/fp:fast>
        )
    endif()

    if(MOTIFCL_ENABLE_NATIVE_ARCH AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        target_compile_options(${target_name} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>
            $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
        )
    endif()

    if(MOTIFCL_ANDROID_MI9_TUNING AND CMAKE_SYSTEM_NAME STREQUAL "Android" AND CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        target_compile_options(${target_name} PRIVATE
            $<$<CXX_COMPILER_ID:Clang>:-mcpu=cortex-a76>
        )
    endif()
endfunction()
