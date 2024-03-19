file(READ sizes SIZES)

list(GET SIZES 0 CORPLEN)
list(GET SIZES 1 VOCABLEN)

configure_file(
    ${INPUT_FILE}
    ${OUTPUT_FILE}
    @ONLY
)