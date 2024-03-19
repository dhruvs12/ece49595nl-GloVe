#include <iostream>
#include <cstdint>
#include <fstream>
#include <array>
#include <vector>
#include "filenames.hpp"
#include "glove_types.hpp"


int main(int argc, char **argv) {

    std::array<cooccur_t, 3> arr;

    std::ifstream in_coocurrence_file (cooccurrence_file);

    for (int i = 0; i < 3; i++) {
        in_coocurrence_file.read(reinterpret_cast<char*>(&arr[i]), sizeof(cooccur_t));
    }
    
    in_coocurrence_file.close();


    return EXIT_SUCCESS;
}
