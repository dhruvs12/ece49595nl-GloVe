#include <unordered_map>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "USAGE gen_frequencies corpus.txt output.txt min_count\n";
        return EXIT_FAILURE;
    }

    std::unordered_map<std::string, uint32_t> vocab;

    std::ifstream infile (argv[1]);
    
    if (!infile.is_open()) {
        std::cerr << "Corpus not read properly. Exiting...\n";
        return EXIT_FAILURE;
    }

    std::string word;
    while (infile >> word) {
        vocab[word]++;
    }

    std::vector<std::pair<std::string, uint32_t>> out_vector;
    std::copy(vocab.begin(), vocab.end(), std::back_inserter(out_vector));

    std::sort(out_vector.begin(), out_vector.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    std::ofstream outfile (argv[2]);

    if (!outfile.is_open()) {
        std::cerr << "Output not opened properly. Exiting...\n";
        return EXIT_FAILURE;
    }

    uint32_t min_count = atoi(argv[3]);

    for (auto &[word, freq] : out_vector) {
        if (freq >= min_count)
            outfile << word << " " << freq << "\n";
    }
    
    infile.close();
    outfile.close();

    return EXIT_SUCCESS;
}