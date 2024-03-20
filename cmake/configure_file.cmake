file(READ sizes SIZES)

if (NOT TRAIN)
    list(GET SIZES 0 CORPLEN)
    list(GET SIZES 1 VOCABLEN)
endif()

configure_file(
    ${INPUT_FILE}
    ${OUTPUT_FILE}
    @ONLY
)