#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include "filenames.hpp"
#include "cooccurrence_const.hpp"
#include "training_sizes.hpp"
#include "glove_types.hpp"

// Create vocabulary dictionary
static bool create_index_map(std::unordered_map<std::string, idx_t>& indices, std::string filename) {
    std::ifstream vocab_in_file(filename);
    
    if (!vocab_in_file.is_open()) {
        return false;
    }

    std::string in_string;
    uint32_t freq;
    idx_t i = 0;

    while (vocab_in_file >> in_string >> freq) {
        indices[in_string] = i++;
    }

    vocab_in_file.close();
    return true;
}

static bool create_corpus(std::array<std::string, corplen> &corpus, std::string filename) {
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
    std::unordered_map<std::string, idx_t> indices;

    if (!create_index_map(indices, vocab_file)) {
        std::cerr << std::format("ERROR: {} not opened. Exiting...\n", vocab_file);
        return EXIT_FAILURE;
    }

    std::array<std::string, corplen> corpus;

    if (!create_corpus(corpus, corpus_file)) {
        std::cerr << std::format("ERROR: {} not opened. Exiting...\n", corpus_file);
        return EXIT_FAILURE;
    }

    std::map<cooccur_key_t, cooccur_value_t> cooccurrences;

    // Left include
    idx_t i_idx;
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
    std::ofstream cooccurrence_out_file (cooccurrence_file, std::ios::binary);
    if (!cooccurrence_out_file.is_open()) {
        std::cerr << std::format("ERROR: {} not opened. Exiting...\n", cooccurrence_file);
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