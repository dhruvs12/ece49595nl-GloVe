#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "filenames.hpp"
#include "freq_const.hpp"
#include "glove_types.hpp"

int main() {
    std::unordered_map<std::string, idx_t> vocab;

    std::ifstream infile (corpus_file);
    
    if (!infile.is_open()) {
        std::cerr << std::format("ERROR: {} not read properly. Exiting...\n", corpus_file);
        return EXIT_FAILURE;
    }

    std::string word;
    idx_t i = 0;
    while (infile >> word) {
        i++;
        vocab[word]++;
    }

    std::vector<std::pair<std::string, idx_t>> out_vector;
    std::copy(vocab.begin(), vocab.end(), std::back_inserter(out_vector));

    std::sort(out_vector.begin(), out_vector.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    std::ofstream outfile (vocab_file);

    if (!outfile.is_open()) {
        std::cerr << std::format("ERROR: {} not opened properly. Exiting...\n", vocab_file);
        return EXIT_FAILURE;
    }

    idx_t j = 0;
    for (auto &[word, freq] : out_vector) {
        if (freq >= min_count) {
            outfile << word << " " << freq << "\n";
            j++;
            if (max_size and j >= max_size) break;
        }
    }
    
    infile.close();
    outfile.close();

    std::cout << std::format("{};{}", i, j);

    return EXIT_SUCCESS;
}