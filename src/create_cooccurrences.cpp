#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_map>

// Create vocabulary dictionary

#define CORPLEN 22
#define VOCABLEN 13

#ifndef LSIZE
#define LSIZE 3
#endif

#ifndef SYMMETRIC
#define SYMMETRIC 1
#endif

constexpr int corplen = CORPLEN;
constexpr int vocablen = VOCABLEN;
constexpr int lsize = LSIZE;
constexpr bool symmetric = SYMMETRIC;

using cooccur_idx_t = uint32_t;
using cooccur_key_t = std::pair<cooccur_idx_t, cooccur_idx_t>;
using cooccur_value_t = long double;
using cooccur_map_iter_t = std::pair<cooccur_key_t, cooccur_value_t>;

bool create_index_map(std::unordered_map<std::string, cooccur_idx_t>& indices, std::string filename) {
    std::ifstream vocab_in_file(filename);
    
    if (!vocab_in_file.is_open()) {
        return false;
    }

    std::string in_string;
    uint32_t freq;
    cooccur_idx_t i = 0;

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
    uint32_t i = 0;

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

    std::unordered_map<std::string, cooccur_idx_t> indices;

    if (!create_index_map(indices, argv[1])) {
        std::cerr << "ERROR: vocab.txt not opened. Exiting...\n";
        return EXIT_FAILURE;
    }

    std::array<std::string, corplen> corpus;

    if (!create_corpus(corpus, argv[2])) {
        std::cerr << "ERROR: corpus.txt not opened. Exiting...\n";
        return EXIT_FAILURE;
    }

    std::map<cooccur_key_t, cooccur_value_t> cooccurrences;

    // Left include
    cooccur_idx_t i_idx;
    for (int i = 0; i < lsize; i++) {
        if (indices.find(corpus[i]) != indices.end()) {
            i_idx = indices[corpus[i]];
            for (int j = 0; j < i; j++) {
                if (indices.find(corpus[j]) != indices.end()) {
                    cooccurrences[{i_idx, indices[corpus[j]]}] += (1. / (abs(i - j)));
                    if (symmetric) cooccurrences[{indices[corpus[j]], i_idx}] += (1. / (abs(i - j)));
                }
            }
        }
    }

    // Inner
    for (int i = lsize; i < corplen; i++) {
        if (indices.find(corpus[i]) != indices.end()) {
            i_idx = indices[corpus[i]];
            for (int j = i - lsize; j < i; j++) {
                if (indices.find(corpus[j]) != indices.end()) {
                    cooccurrences[{i_idx, indices[corpus[j]]}] += (1. / (abs(i - j)));
                    if (symmetric) cooccurrences[{indices[corpus[j]], i_idx}] += (1. / (abs(i - j)));
                }
            }
        }
    }

    std::vector<cooccur_map_iter_t> shuffled;
    std::copy(cooccurrences.begin(), cooccurrences.end(), std::back_inserter(shuffled));

    // Shuffle cooccurrences
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled.begin(), shuffled.end(), g);

    // Write to file
    std::ofstream cooccurrence_out_file (argv[3], std::ios::binary);
    if (!cooccurrence_out_file.is_open()) {
        std::cerr << "ERROR: output_file.bin not opened. Exiting...\n";
        return EXIT_FAILURE;
    }

    for (auto &[words, freq] : shuffled) {
        cooccurrence_out_file.write(reinterpret_cast<const char *>(&words.first), sizeof(words.first));
        cooccurrence_out_file.write(reinterpret_cast<const char *>(&words.second), sizeof(words.second));
        cooccurrence_out_file.write(reinterpret_cast<const char *>(&freq), sizeof(freq));
    
    }

    cooccurrence_out_file.close();

    return EXIT_SUCCESS;
}