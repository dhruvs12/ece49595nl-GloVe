# Build settings
cmake_minimum_required(VERSION 3.27)

project(GloVe LANGUAGES CXX VERSION 0.1)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/GloVe_package_versions.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -Wextra -Wpedantic -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer -march=native -flto=8 -funroll-loops")

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Find linked library tbb
find_package(TBB REQUIRED)

# Configure files
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/dist)
file(READ config.json CONFIG)

function(get_json target_variable usepath)
    string(
        JSON 
        result
        GET 
        ${CONFIG}
        ${ARGN}
    )

    if (usepath)
        get_filename_component(result ${result} ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
    endif()

    set(${target_variable} ${result} PARENT_SCOPE)
endfunction()

get_json(CORPUS_FILE ON filenames corpus_file)
get_json(VOCAB_FILE ON filenames vocab_file)
get_json(COOCCURRENCE_FILE ON filenames cooccurrence_file)
get_json(VECTOR_FILE ON filenames vector_file)
get_json(CONTEXT_FILE ON filenames context_file)
get_json(BIAS_FILE ON filenames bias_file)

get_json(MIN_COUNT OFF frequencies min_count)
get_json(MAX_SIZE OFF frequencies max_size)

get_json(LSIZE OFF cooccurrence window_size)
get_json(SYMMETRIC OFF cooccurrence symmetric)
get_json(DIST_WEIGHTING OFF cooccurrence distance_weighting)

get_json(ETA OFF training params eta)
get_json(ALPHA OFF training params alpha)
get_json(X_MAX OFF training params x_max)
get_json(GRADIENT_CLIP OFF training params gradient_clipping)

get_json(ITERATIONS OFF training runtime iterations)
get_json(NUM_THREADS OFF training runtime num_threads)
get_json(VECTOR_SIZE OFF training runtime vector_size)

set(SRC_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/include")

configure_file("${SRC_INCLUDE_DIR}/freq_const.hpp.in" "${CMAKE_BINARY_DIR}/freq_const.hpp")
configure_file("${SRC_INCLUDE_DIR}/cooccurrence_const.hpp.in" "${CMAKE_BINARY_DIR}/cooccurrence_const.hpp")
configure_file("${SRC_INCLUDE_DIR}/freq_const.hpp.in" "${CMAKE_BINARY_DIR}/freq_const.hpp")
configure_file("${SRC_INCLUDE_DIR}/filenames.hpp.in" "${CMAKE_BINARY_DIR}/filenames.hpp")
configure_file("${SRC_INCLUDE_DIR}/glove_const.hpp.in" "${CMAKE_BINARY_DIR}/glove_const.hpp")

include_directories(${CMAKE_BINARY_DIR})
include_directories("${CMAKE_SOURCE_DIR}/src")

# Generate frequencies
add_executable(frequencies src/gen_frequencies.cpp)

add_custom_target(
    gen_frequencies
    COMMAND frequencies > sizes
    DEPENDS frequencies
    COMMENT "Generating frequencies..."
)

add_custom_command(
    OUTPUT input_size_output
    COMMAND ${CMAKE_COMMAND} -DTRAIN="OFF" -DINPUT_FILE="${SRC_INCLUDE_DIR}/training_sizes.hpp.in" -DOUTPUT_FILE="${CMAKE_BINARY_DIR}/training_sizes.hpp" -P "${CMAKE_SOURCE_DIR}/cmake/configure_file.cmake"
    DEPENDS gen_frequencies
    COMMENT "Writing input sizes..."
)

add_custom_target(size_const DEPENDS input_size_output)
add_executable(cooccurrences src/create_cooccurrences.cpp)
add_dependencies(cooccurrences size_const)

add_custom_target(
    create_cooccurrences
    COMMAND cooccurrences > sizes
    DEPENDS cooccurrences
    COMMENT "Creating cooccurrences..."
)

add_custom_command(
    OUTPUT train_size_output
    COMMAND ${CMAKE_COMMAND} -DTRAIN="ON" -DINPUT_FILE="${SRC_INCLUDE_DIR}/x_size.hpp.in" -DOUTPUT_FILE="${CMAKE_BINARY_DIR}/x_size.hpp" -P "${CMAKE_SOURCE_DIR}/cmake/configure_file.cmake"
    DEPENDS create_cooccurrences
    COMMENT "Writing training size..."
)
add_custom_target(train_const DEPENDS train_size_output)

add_executable(train src/glove.cpp)
add_dependencies(train train_const)
target_link_libraries(train PRIVATE TBB::tbb atomic)

message(STATUS "${DIST_WEIGHTING} weighting")

add_custom_target(
    glove
    ALL
    COMMAND train
    DEPENDS train
    COMMENT "Training model..."
)