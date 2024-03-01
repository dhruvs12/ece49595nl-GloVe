#include <map>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "USAGE gen_frequencies corpus.txt output.txt min_count\n";
        return EXIT_FAILURE;
    }

    std::map<std::string, uint32_t> vocab;

    std::ifstream infile (argv[1]);
    
    if (!infile.is_open()) {
        std::cerr << "Corpus not read properly. Exiting...\n";
        return EXIT_FAILURE;
    }

    std::string word;
    while (infile >> word) {
        vocab[word]++;
    }

    std::ofstream outfile (argv[2]);

    if (!outfile.is_open()) {
        std::cerr << "Output not opened properly. Exiting...\n";
        return EXIT_FAILURE;
    }

    int min_count = atoi(argv[3]);

    for (auto &[word, freq] : vocab) {
        if (freq >= min_count)
            outfile << word << " " << freq << "\n";
    }
    
    infile.close();
    outfile.close();

    return EXIT_SUCCESS;
}