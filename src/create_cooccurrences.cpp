#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <map>
#include <cstdint>
#include <array>
#include <algorithm>
#include <random>

// Create vocabulary dictionary

#define CORPLEN
#define VOCABLEN

#ifndef LSIZE
#define LSIZE 15
#endif

#ifndef RSIZE
#define RSIZE 15
#endif

constexpr int corplen = CORPLEN;
constexpr int vocablen = VOCABLEN;
constexpr int lsize = LSIZE;
constexpr int rsize = RSIZE;

bool create_index_map(std::unordered_map<std::string, int>& indices, std::string filename) {
    std::ifstream vocab_in_file(filename);
    
    if (!vocab_in_file.is_open()) {
        return false;
    }

    std::string in_string;
    int freq, i = 0;

    while (vocab_in_file >> in_string >> freq) {
        indices[in_string] = i++;
    }

    vocab_in_file.close();
    return true;
}

bool create_corpus(std::array<std::string, corplen> &corpus, std::string filename) {
    std::ifstream corpus_in_file(filename);
    
    if (!corpus_in_file.is_open()) {
        return false;
    }

    std::string in_string;
    int i = 0;

    while (corpus_in_file >> in_string) {
        corpus[i++] = in_string;
    }

    corpus_in_file.close();
    return true;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "USAGE: ./create_frequencies vocab.txt corpus.txt output_file.bin\n";
        return EXIT_FAILURE;
    }

    std::unordered_map<std::string, int> indices;

    if (!create_index_map(indices, argv[1])) {
        std::cerr << "ERROR: vocab.txt not opened. Exiting...\n";
        return EXIT_FAILURE;
    }

    std::array<std::string, corplen> corpus;

    if (!create_corpus(corpus, argv[2])) {
        std::cerr << "ERROR: corpus.txt not opened. Exiting...\n";
        return EXIT_FAILURE;
    }

    std::map<std::pair<uint32_t, uint32_t>, int> cooccurrences;


    // Left include
    for (int i = 0; i < lsize; i++) {
        if (indices.find(corpus[i]) != indices.end()) {
            for (int j = 0; j < i; j++) {
                if (indices.find(corpus[j]) != indices.end()) {
                    cooccurrences[{indices[corpus[i]], indices[corpus[j]]}]++;
                }
            }
        }
    }

    // Inner
    for (int i = lsize; i < corplen - rsize; i++) {
        if (indices.find(corpus[i]) != indices.end()) {
            for (int j = i - lsize; j < i + rsize; j++) {
                if (indices.find(corpus[j]) != indices.end()) {
                    cooccurrences[{indices[corpus[i]], indices[corpus[j]]}]++;
                }
            }
        }
    }

    // Right include
    for (int i = corplen - rsize; i < corplen; i++) {
        if (indices.find(corpus[i]) != indices.end()) {
            for (int j = i - lsize; j < corplen; j++) {
                if (indices.find(corpus[j]) != indices.end()) {
                    cooccurrences[{indices[corpus[i]], indices[corpus[j]]}]++;
                }
            }
        }
    }

    // Shuffle cooccurrences
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(cooccurrences.begin(), cooccurrences.end(), g);


    // Write to file
    std::ofstream cooccurrence_out_file (argv[3], std::ios::binary);
    if (!cooccurrence_out_file.is_open()) {
        std::cerr << "ERROR: output_file.bin not opened. Exiting...\n";
        return EXIT_FAILURE;
    }

    for (auto &[words, freq] : cooccurrences) {
        if (words.first != words.second) {
            cooccurrence_out_file.write(reinterpret_cast<const char *>(&words.first), sizeof(int));
            cooccurrence_out_file.write(reinterpret_cast<const char *>(&words.second), sizeof(int));
            cooccurrence_out_file.write(reinterpret_cast<const char *>(&freq), sizeof(int));
        }
    }

    cooccurrence_out_file.close();

    return EXIT_SUCCESS;
}