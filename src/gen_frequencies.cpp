#include <unordered_map>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include "include/freq_const.hpp"
#include "include/glove_types.hpp"

constexpr idx_t min_count = MIN_COUNT;
constexpr idx_t max_size = MAX_SIZE;

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "USAGE gen_frequencies corpus.txt output.txt min_count\n";
        return EXIT_FAILURE;
    }

    std::unordered_map<std::string, idx_t> vocab;

    std::ifstream infile (argv[1]);
    
    if (!infile.is_open()) {
        std::cerr << "Corpus not read properly. Exiting...\n";
        return EXIT_FAILURE;
    }

    std::string word;
    while (infile >> word) {
        vocab[word]++;
    }

    std::vector<std::pair<std::string, idx_t>> out_vector;
    std::copy(vocab.begin(), vocab.end(), std::back_inserter(out_vector));

    std::sort(out_vector.begin(), out_vector.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    std::ofstream outfile (argv[2]);

    if (!outfile.is_open()) {
        std::cerr << "Output not opened properly. Exiting...\n";
        return EXIT_FAILURE;
    }

    idx_t i = 0;
    for (auto &[word, freq] : out_vector) {
        if (freq >= min_count) {
            outfile << word << " " << freq << "\n";
            i++;
            if (i >= max_size) break;
        }
    }
    
    infile.close();
    outfile.close();

    return EXIT_SUCCESS;
}