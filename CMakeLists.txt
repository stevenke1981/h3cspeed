cmake_minimum_required(VERSION 3.25)

project(h3cspeed VERSION 0.2.0 LANGUAGES C CXX)

include(FetchContent)
include(CheckLanguage)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(H3CSPEED_OVERLAY_TESTS_ONLY "Configure portable overlay tests without CUDA/upstream downloads" OFF)
option(H3CSPEED_ENABLE_INT8 "Enable CUDA int8 weight paths" ON)
option(H3CSPEED_ENABLE_TESTS "Build h3cspeed tests" ON)
option(H3CSPEED_ENABLE_SANITIZERS "Enable host Address/Undefined sanitizers in non-CUDA tests" OFF)
option(H3CSPEED_WERROR "Treat overlay warnings as errors" OFF)
option(H3CSPEED_FAST_MATH "Enable CUDA --use_fast_math (faster, lower numerical fidelity)" OFF)
set(H3CSPEED_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures, e.g. native, 86 for RTX 3070 Ti/3090, or 86-real;120")
set(H3CSPEED_H3_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/h3" CACHE PATH
    "Prepared antirez/h3.c source directory")
set(H3CSPEED_ICU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/icu" CACHE PATH
    "ICU4C root (Windows builds may use the official prebuilt archive)")

if(H3CSPEED_OVERLAY_TESTS_ONLY)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    enable_testing()
    add_executable(test_resize tests/test_resize.c src/h3_resize_portable.c)
    add_executable(test_offload_policy tests/test_offload_policy.c
        src/h3_offload_policy.c)
    add_executable(test_profile_report tests/test_profile_report.c
        src/h3_profile.c)
    target_include_directories(test_profile_report PRIVATE src)
    target_include_directories(test_offload_policy PRIVATE src)
    if(MSVC)
        target_compile_definitions(test_offload_policy PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_offload_policy PRIVATE
            /W4 /WX /permissive-)
    else()
        target_compile_options(test_offload_policy PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror)
    endif()
    if(MSVC)
        target_compile_definitions(test_profile_report PRIVATE
            _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
        target_compile_options(test_profile_report PRIVATE
            /W4 /WX /permissive-)
    else()
        target_compile_options(test_profile_report PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wconversion
            -Wsign-conversion -Werror)
    endif()
    target_include_directories(test_resize PRIVATE src)
    if(MSVC)
        target_compile_definitions(test_resize PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_resize PRIVATE /W4 /permissive-)
    else()
        target_link_libraries(test_resize PRIVATE m)
        target_compile_options(test_resize PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME portable_resize COMMAND test_resize)
    add_test(NAME offload_policy COMMAND test_offload_policy)
    add_test(NAME profile_report_schema
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_profile_report.py
                $<TARGET_FILE:test_profile_report>)
    add_test(NAME perf002_manifest
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_perf002_manifest.py)
    add_test(NAME perf002_media_harness
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_perf002_runner.py)
    add_test(NAME online_softmax_math
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_online_softmax.py)
    add_test(NAME quantized_sidecar_v2
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quantized_sidecar_v2.py)
    add_test(NAME alias_free_snake_math
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_alias_free_snake.py)
    add_test(NAME project_metadata
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_project_metadata.py)
    add_test(NAME bootstrap_tools
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bootstrap_tools.py)
    add_test(NAME low_vram_static
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_low_vram_backend.py)
    add_test(NAME wrapper_profile
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_3070ti_wrapper.py)
    add_test(NAME quantized_wrapper
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quantized_wrapper.py)
    add_test(NAME quantized_60s_runner
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quantized_60s_runner.py)
    add_test(NAME runtime_package
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime_package.py)
    message(STATUS "h3cspeed: configured portable overlay tests only")
    return()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bootstrap.py
            --project-root ${CMAKE_CURRENT_SOURCE_DIR} --verify-only
            --source-dir ${H3CSPEED_H3_SOURCE_DIR}
    RESULT_VARIABLE H3CSPEED_UPSTREAM_VERIFY_RESULT
    OUTPUT_VARIABLE H3CSPEED_UPSTREAM_VERIFY_OUTPUT
    ERROR_VARIABLE H3CSPEED_UPSTREAM_VERIFY_ERROR
)
if(NOT H3CSPEED_UPSTREAM_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Prepared h3.c source verification failed:\n"
        "${H3CSPEED_UPSTREAM_VERIFY_OUTPUT}${H3CSPEED_UPSTREAM_VERIFY_ERROR}")
endif()

enable_language(CUDA)

if(NOT EXISTS "${H3CSPEED_H3_SOURCE_DIR}/.h3cspeed-pinned-revision")
    message(FATAL_ERROR
        "Pinned h3.c sources are not prepared. Run: python3 scripts/bootstrap.py")
endif()

set(CMAKE_CUDA_ARCHITECTURES "${H3CSPEED_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
if(WIN32)
    find_path(ICU_INCLUDE_DIR unicode/uchar.h
        HINTS "${H3CSPEED_ICU_ROOT}/include" REQUIRED)
    find_library(ICU_UC_LIBRARY NAMES icuuc icuuc76
        HINTS "${H3CSPEED_ICU_ROOT}/lib64" REQUIRED)
    find_file(ICU_UC_RUNTIME NAMES icuuc76.dll
        HINTS "${H3CSPEED_ICU_ROOT}/bin64" REQUIRED)
    find_file(ICU_DATA_RUNTIME NAMES icudt76.dll
        HINTS "${H3CSPEED_ICU_ROOT}/bin64" REQUIRED)
    add_library(H3CSPEED::ICU UNKNOWN IMPORTED)
    set_target_properties(H3CSPEED::ICU PROPERTIES
        IMPORTED_LOCATION "${ICU_UC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ICU_INCLUDE_DIR}")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ICU REQUIRED IMPORTED_TARGET icu-uc)
    add_library(H3CSPEED::ICU ALIAS PkgConfig::ICU)
endif()

# yyjson is C, fast on the released tokenizer.json, and avoids bringing a C++
# JSON dependency into the public H3 C API.
FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    # Immutable upstream commit; do not replace this with a moving release tag.
    GIT_TAG 9ddba001a4ea88e93b46932e5c5b87b222e19a5f
    GIT_SHALLOW FALSE
)
set(YYJSON_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YYJSON_BUILD_MISC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yyjson)

set(H3_UPSTREAM_C
    h3.c
    h3_host.c
    h3_safetensors.c
    h3_weights.c
    h3_text_encoder.c
    h3_dit_schedule.c
    h3_dit.c
    h3_video_vae.c
    h3_video_encoder.c
    h3_audio_vae.c
    h3_ffmpeg.c
    h3_terminal.c
    h3_vision_encoder.c
    h3_multimodal.c
)
list(TRANSFORM H3_UPSTREAM_C PREPEND "${H3CSPEED_H3_SOURCE_DIR}/")

set(H3CSPEED_CUDA_SOURCES
    src/h3_cuda_probe.cu
    src/h3_gpu_cuda.cu
    src/h3_gpu_cuda_attention.cu
    src/h3_gpu_cuda_vae.cu
)

add_library(h3cspeed_core STATIC
    ${H3_UPSTREAM_C}
    src/h3_text_embedding_file.c
    src/h3_fl2va_sidecar.c
    src/h3_tokenizer_portable.c
    src/h3_resize_portable.c
    src/h3_offload_policy.c
    src/h3_profile.c
    $<$<PLATFORM_ID:Windows>:src/h3_windows_posix.c>
    ${H3CSPEED_CUDA_SOURCES}
)

# ICU exposes conversion-heavy macros; keep the project strict without turning
# third-party macro expansion into h3cspeed build failures.
if(NOT MSVC)
    set_source_files_properties(src/h3_tokenizer_portable.c PROPERTIES
        COMPILE_OPTIONS "-Wno-conversion"
    )
endif()

set_target_properties(h3cspeed_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME h3cspeed
)

target_include_directories(h3cspeed_core
    PUBLIC
        $<BUILD_INTERFACE:${H3CSPEED_H3_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(h3cspeed_core PRIVATE
    _GNU_SOURCE
    H3CSPEED_CUDA=1
    H3CSPEED_ENABLE_INT8=$<BOOL:${H3CSPEED_ENABLE_INT8}>
)

if(MSVC)
    target_compile_options(h3cspeed_core PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:/O2;/W4;/permissive-;/D_CRT_SECURE_NO_WARNINGS>
        $<$<COMPILE_LANGUAGE:CUDA>:-O2;--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=/W4;-Xcompiler=/D_CRT_SECURE_NO_WARNINGS>
    )
    target_compile_options(h3cspeed_core PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:/FI${CMAKE_CURRENT_SOURCE_DIR}/src/h3_msvc_compat.h>
    )
else()
    target_compile_options(h3cspeed_core PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-O3;-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion;-Wno-sign-conversion>
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion;-Wno-sign-conversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
    )
endif()
if(H3CSPEED_WERROR)
    if(MSVC)
        target_compile_options(h3cspeed_core PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:/WX>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
    else()
        target_compile_options(h3cspeed_core PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()
endif()
if(H3CSPEED_FAST_MATH)
    target_compile_options(h3cspeed_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

target_link_libraries(h3cspeed_core
    PUBLIC
        Threads::Threads
        H3CSPEED::ICU
        yyjson
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        $<$<PLATFORM_ID:Windows>:advapi32>
        $<$<NOT:$<PLATFORM_ID:Windows>>:m>
        $<$<NOT:$<PLATFORM_ID:Windows>>:dl>
)

add_executable(h3cspeed
    "${H3CSPEED_H3_SOURCE_DIR}/main.c"
    "${H3CSPEED_H3_SOURCE_DIR}/h3_cli.c"
)
if(WIN32)
    target_sources(h3cspeed PRIVATE src/linenoise_windows.c)
else()
    target_sources(h3cspeed PRIVATE "${H3CSPEED_H3_SOURCE_DIR}/linenoise.c")
endif()
target_include_directories(h3cspeed PRIVATE
    "${H3CSPEED_H3_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_options(h3cspeed PRIVATE
    $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O3;-Wall;-Wextra;-Wpedantic;-Wno-conversion;-Wno-variadic-macros>
    $<$<C_COMPILER_ID:MSVC>:/O2;/W4;/D_CRT_SECURE_NO_WARNINGS;/FI${CMAKE_CURRENT_SOURCE_DIR}/src/h3_msvc_compat.h>
)
target_link_libraries(h3cspeed PRIVATE h3cspeed_core)
set_target_properties(h3cspeed PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(h3cspeed-cuda-info src/h3_cuda_info_main.c)
target_include_directories(h3cspeed-cuda-info PRIVATE
    "${H3CSPEED_H3_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
if(MSVC)
    target_compile_options(h3cspeed-cuda-info PRIVATE
        /D_CRT_SECURE_NO_WARNINGS /FI${CMAKE_CURRENT_SOURCE_DIR}/src/h3_msvc_compat.h)
endif()
target_link_libraries(h3cspeed-cuda-info PRIVATE h3cspeed_core)
set_target_properties(h3cspeed-cuda-info PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(WIN32)
    foreach(runtime_target h3cspeed h3cspeed-cuda-info)
        add_custom_command(TARGET ${runtime_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ICU_UC_RUNTIME}" "$<TARGET_FILE_DIR:${runtime_target}>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ICU_DATA_RUNTIME}" "$<TARGET_FILE_DIR:${runtime_target}>"
            VERBATIM)
    endforeach()
endif()

if(H3CSPEED_ENABLE_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    enable_testing()
    add_executable(test_resize tests/test_resize.c src/h3_resize_portable.c)
    target_include_directories(test_resize PRIVATE src)
    add_executable(test_cuda_scale_add tests/test_cuda_scale_add.c)
    target_include_directories(test_cuda_scale_add PRIVATE
        "${H3CSPEED_H3_SOURCE_DIR}")
    target_link_libraries(test_cuda_scale_add PRIVATE h3cspeed_core)
    set_target_properties(test_cuda_scale_add PROPERTIES
        LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_executable(test_cuda_numeric_regression
        tests/test_cuda_numeric_regression.c)
    target_include_directories(test_cuda_numeric_regression PRIVATE
        "${H3CSPEED_H3_SOURCE_DIR}")
    target_link_libraries(test_cuda_numeric_regression PRIVATE h3cspeed_core)
    set_target_properties(test_cuda_numeric_regression PROPERTIES
        LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_executable(bench_cuda_attention tests/bench_cuda_attention.c)
    target_include_directories(bench_cuda_attention PRIVATE
        "${H3CSPEED_H3_SOURCE_DIR}")
    target_link_libraries(bench_cuda_attention PRIVATE h3cspeed_core)
    set_target_properties(bench_cuda_attention PROPERTIES
        LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_executable(test_cuda_convrot_numeric
        tests/test_cuda_convrot_numeric.c)
    target_include_directories(test_cuda_convrot_numeric PRIVATE
        src "${H3CSPEED_H3_SOURCE_DIR}")
    target_link_libraries(test_cuda_convrot_numeric PRIVATE h3cspeed_core)
    set_target_properties(test_cuda_convrot_numeric PROPERTIES
        LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_executable(test_cuda_nvfp4_numeric
        tests/test_cuda_nvfp4_numeric.c)
    target_include_directories(test_cuda_nvfp4_numeric PRIVATE
        src "${H3CSPEED_H3_SOURCE_DIR}")
    target_link_libraries(test_cuda_nvfp4_numeric PRIVATE h3cspeed_core)
    set_target_properties(test_cuda_nvfp4_numeric PROPERTIES
        LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_executable(test_cuda_vae_layer_major_parity
        tests/test_cuda_vae_layer_major_parity.c)
    target_include_directories(test_cuda_vae_layer_major_parity PRIVATE
        "${H3CSPEED_H3_SOURCE_DIR}")
    target_link_libraries(test_cuda_vae_layer_major_parity PRIVATE
        h3cspeed_core)
    set_target_properties(test_cuda_vae_layer_major_parity PROPERTIES
        LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_executable(test_text_prompt_hash tests/test_text_prompt_hash.c)
    target_include_directories(test_text_prompt_hash PRIVATE
        "${H3CSPEED_H3_SOURCE_DIR}")
    target_link_libraries(test_text_prompt_hash PRIVATE h3cspeed_core)
    set_target_properties(test_text_prompt_hash PROPERTIES
        LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_executable(test_text_embedding_file tests/test_text_embedding_file.c)
    target_include_directories(test_text_embedding_file PRIVATE
        src "${H3CSPEED_H3_SOURCE_DIR}")
    target_link_libraries(test_text_embedding_file PRIVATE h3cspeed_core)
    set_target_properties(test_text_embedding_file PROPERTIES
        LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    if(WIN32)
        add_custom_command(TARGET test_text_prompt_hash POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ICU_UC_RUNTIME}" "$<TARGET_FILE_DIR:test_text_prompt_hash>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ICU_DATA_RUNTIME}" "$<TARGET_FILE_DIR:test_text_prompt_hash>"
            VERBATIM)
    endif()
    add_executable(test_offload_policy tests/test_offload_policy.c
        src/h3_offload_policy.c)
    target_include_directories(test_offload_policy PRIVATE src)
    if(MSVC)
        target_compile_definitions(test_offload_policy PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_definitions(test_resize PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_offload_policy PRIVATE /W4 /WX /permissive-)
        target_compile_options(test_resize PRIVATE /W4 /permissive-)
        target_compile_definitions(test_cuda_scale_add PRIVATE
            _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_cuda_scale_add PRIVATE /W4 /WX /permissive-)
        target_compile_definitions(test_cuda_numeric_regression PRIVATE
            _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_cuda_numeric_regression PRIVATE
            /W4 /WX /permissive-)
        target_compile_definitions(bench_cuda_attention PRIVATE
            _CRT_SECURE_NO_WARNINGS)
        target_compile_options(bench_cuda_attention PRIVATE
            /W4 /WX /permissive-)
        target_compile_definitions(test_cuda_convrot_numeric PRIVATE
            _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_cuda_convrot_numeric PRIVATE
            /W4 /WX /permissive-)
        target_compile_definitions(test_cuda_nvfp4_numeric PRIVATE
            _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_cuda_nvfp4_numeric PRIVATE
            /W4 /WX /permissive-)
        target_compile_definitions(test_text_prompt_hash PRIVATE
            _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_text_prompt_hash PRIVATE
            /W4 /WX /permissive- /FI${CMAKE_CURRENT_SOURCE_DIR}/src/h3_msvc_compat.h)
        target_compile_definitions(test_text_embedding_file PRIVATE
            _CRT_SECURE_NO_WARNINGS)
        target_compile_options(test_text_embedding_file PRIVATE
            /W4 /WX /permissive- /FI${CMAKE_CURRENT_SOURCE_DIR}/src/h3_msvc_compat.h)
    else()
        target_compile_options(test_offload_policy PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror)
        target_link_libraries(test_resize PRIVATE m)
        target_compile_options(test_resize PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(test_cuda_scale_add PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(test_cuda_numeric_regression PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(bench_cuda_attention PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(test_cuda_convrot_numeric PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(test_cuda_nvfp4_numeric PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(test_cuda_vae_layer_major_parity PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(test_text_embedding_file PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
    if(H3CSPEED_ENABLE_SANITIZERS)
        target_compile_options(test_resize PRIVATE -fsanitize=address,undefined)
        target_link_options(test_resize PRIVATE -fsanitize=address,undefined)
    endif()
    add_test(NAME portable_resize COMMAND test_resize)
    add_test(NAME offload_policy COMMAND test_offload_policy)
    add_test(NAME cuda_scale_add COMMAND test_cuda_scale_add)
    add_test(NAME text_embedding_file COMMAND test_text_embedding_file)
    set_tests_properties(cuda_scale_add PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME cuda_numeric_regression COMMAND test_cuda_numeric_regression)
    set_tests_properties(cuda_numeric_regression PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME cuda_convrot_numeric COMMAND test_cuda_convrot_numeric)
    set_tests_properties(cuda_convrot_numeric PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME cuda_nvfp4_numeric COMMAND test_cuda_nvfp4_numeric)
    set_tests_properties(cuda_nvfp4_numeric PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME cuda_vae_layer_major_parity
        COMMAND test_cuda_vae_layer_major_parity)
    set_tests_properties(cuda_vae_layer_major_parity PROPERTIES
        SKIP_RETURN_CODE 77)
    add_test(NAME online_softmax_math
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_online_softmax.py)
    add_test(NAME alias_free_snake_math
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_alias_free_snake.py)
    add_test(NAME project_metadata
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_project_metadata.py)
    add_test(NAME bootstrap_tools
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bootstrap_tools.py)
    add_test(NAME low_vram_static
        COMMAND ${CMAKE_COMMAND} -E env
            H3CSPEED_CLI=$<TARGET_FILE:h3cspeed>
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_low_vram_backend.py)
    add_test(NAME wrapper_profile
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_3070ti_wrapper.py)
    add_test(NAME quantized_wrapper
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quantized_wrapper.py)
    add_test(NAME quantized_60s_runner
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quantized_60s_runner.py)
    add_test(NAME runtime_package
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime_package.py)
    if(WIN32)
        add_executable(test_windows_child tests/test_windows_child.c)
        add_executable(test_windows_compat tests/test_windows_compat.c
            src/h3_windows_posix.c)
        add_executable(test_ffmpeg_bridge tests/test_ffmpeg_bridge.c)
        target_include_directories(test_windows_compat PRIVATE src)
        target_include_directories(test_ffmpeg_bridge PRIVATE
            src "${H3CSPEED_H3_SOURCE_DIR}")
        target_compile_options(test_windows_compat PRIVATE
            /W4 /WX /permissive-
            /D_CRT_SECURE_NO_WARNINGS
            /FI${CMAKE_CURRENT_SOURCE_DIR}/src/h3_msvc_compat.h)
        target_compile_options(test_ffmpeg_bridge PRIVATE
            /W4 /WX /permissive-
            /D_CRT_SECURE_NO_WARNINGS
            /FI${CMAKE_CURRENT_SOURCE_DIR}/src/h3_msvc_compat.h)
        target_link_libraries(test_ffmpeg_bridge PRIVATE h3cspeed_core)
        set_target_properties(test_ffmpeg_bridge PROPERTIES
            LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        target_compile_options(test_windows_child PRIVATE /W4 /WX /permissive-)
        add_test(NAME windows_compat
            COMMAND test_windows_compat $<TARGET_FILE:test_windows_child>)
        add_test(NAME ffmpeg_bridge COMMAND test_ffmpeg_bridge)
    endif()
endif()

install(TARGETS h3cspeed h3cspeed-cuda-info h3cspeed_core
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES
    "${H3CSPEED_H3_SOURCE_DIR}/h3.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/h3cspeed
)
if(WIN32)
    install(FILES "${ICU_UC_RUNTIME}" "${ICU_DATA_RUNTIME}"
        DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(PROGRAMS scripts/run-3070ti-8gb.sh
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    RENAME h3cspeed-3070ti-8gb
)
install(PROGRAMS scripts/smoke-3070ti-8gb.sh
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    RENAME h3cspeed-3070ti-smoke
)
install(PROGRAMS scripts/fast-quality-3070ti-8gb.sh
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    RENAME h3cspeed-3070ti-fast-quality
)
install(FILES
    profiles/rtx3070ti-8gb.env
    profiles/rtx3070ti-fast-quality.env
    DESTINATION ${CMAKE_INSTALL_DATADIR}/h3cspeed/profiles
)
