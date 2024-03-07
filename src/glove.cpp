#include <iostream>
#include <cstdint>
#include <fstream>
#include <array>
#include <vector>

// #define CORPLEN
// #define VOCABLEN

// constexpr int corplen = CORPLEN;
// constexpr int vocablen = VOCABLEN;

typedef struct _cooccur_t {
    uint32_t token1;
    uint32_t token2;
    uint32_t val;
} cooccur_t;


int main(int argc, char **argv) {

    std::array<cooccur_t, 3> arr;

    for (int i = 0; i < 3; i++) {
        in_coocurrence_file.read(reinterpret_cast<char*>(&arr[i]), sizeof(cooccur_t));
    }
    
    in_coocurrence_file.close();


    return EXIT_SUCCESS;
}
