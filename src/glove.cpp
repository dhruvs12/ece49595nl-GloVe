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

    std::vector<std::vector<uint32_t>> v {
        { 1, 2, 0xFFF},
        { 5, 6, 0xFFFF},
        { 7, 8, 0xFFFFF}
    };

    std::ofstream out_occurrence_file ("file.bin", std::ios::binary);
    if (!out_occurrence_file.is_open()) {
        std::cerr << "NO\n";
        return EXIT_FAILURE;
    }

    for (auto x : v) {
        for (auto y : x) {
            out_occurrence_file.write(reinterpret_cast<const char *>(&y), sizeof(uint32_t));
        }
    }

    out_occurrence_file.close();

    std::ifstream in_coocurrence_file ("file.bin", std::ios::binary);

    if (!in_coocurrence_file.is_open()) {
        std::cerr << "ERROR: coocurrence file not opened. Exiting...\n";
        return EXIT_FAILURE;
    }

    std::array<cooccur_t, 3> arr;

    for (int i = 0; i < 3; i++) {
        in_coocurrence_file.read(reinterpret_cast<char*>(&arr[i]), sizeof(cooccur_t));
    }

    for (auto &zz : arr) {
        std::cout << zz.token1 << " " << zz.token2 << " " << zz.val << "\n";
    }
    
    in_coocurrence_file.close();

    return EXIT_SUCCESS;
}
