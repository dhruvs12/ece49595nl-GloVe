#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "cooccurrence_const.hpp"
#include "filenames.hpp"
#include "glove_types.hpp"
#include "training_sizes.hpp"

// Create vocabulary dictionary
static bool create_index_map(std::unordered_map<std::string, idx_t>& indices, std::string filename) {
    std::ifstream vocab_in_file(filename);
    
    if (!vocab_in_file.is_open()) {
        return false;
    }

    std::string in_string;
    idx_t i = 0, freq;

    while (vocab_in_file >> in_string >> freq) {
        indices[in_string] = i++;
    }

    vocab_in_file.close();
    return true;
}

static bool create_corpus(std::vector<std::string> &corpus, std::string filename) {
    std::ifstream corpus_in_file(filename);
    
    if (!corpus_in_file.is_open()) {
        return false;
    }

    std::string in_string;
    idx_t i = 0;

    while (corpus_in_file >> in_string) {
        corpus[i++] = in_string;
    }

    corpus_in_file.close();
    return true;
}

int main() {
    std::unordered_map<std::string, idx_t> indices;
    indices.reserve(vocablen);

    if (!create_index_map(indices, vocab_file)) {
        std::cerr << std::format("ERROR: {} not opened. Exiting...\n", vocab_file);
        return EXIT_FAILURE;
    }

    std::vector<std::string> corpus (corplen);

    if (!create_corpus(corpus, corpus_file)) {
        std::cerr << std::format("ERROR: {} not opened. Exiting...\n", corpus_file);
        return EXIT_FAILURE;
    }
    
    std::unordered_map<cooccur_key_t, cooccur_value_t, boost::hash<cooccur_key_t>> cooccurrences;
    cooccurrences.reserve(vocablen * vocablen);

    std::vector<idx_t> history (lsize);
    decltype(indices)::iterator i_it;
    idx_t place, i_idx, i = 0;
    cooccur_value_t res;

    // covering filling history and i == lsize case (compared with zero)
    for (place = 0; place < corplen and i <= lsize; place++) {
        if ((i_it = indices.find(corpus[place])) != indices.end()) {
            i_idx = i_it->second;
            idx_t j = i;
            
            if (i != 0) do {
                j--;
                res = (1. / (i - j));
                cooccurrences[{history[j % lsize], i_idx}] += res;
                if (symmetric) cooccurrences[{i_idx, history[j % lsize]}] += res;
            } while (j != 0);

            history[i % lsize] = i_idx;
            i++;
        }
    }

    // Rest of cooccurrences
    for (; place < corplen; place++) {
        if ((i_it = indices.find(corpus[place])) != indices.end()) {
            i_idx = i_it->second;
            for (idx_t j = i - 1; j >= i - lsize; j--) {
                res = (1. / (i - j));
                cooccurrences[{history[j % lsize], i_idx}] += res;
                if (symmetric) cooccurrences[{i_idx, history[j % lsize]}] += res;
            }

            history[i % lsize] = i_idx;
            i++;
        }
    }

    // Copy map to vector
    std::vector<cooccur_map_iter_t> shuffled;
    std::copy(cooccurrences.begin(), cooccurrences.end(), std::back_inserter(shuffled));
    
    // Shuffle cooccurrences
    static std::random_device rd;
    static std::mt19937 g(rd());
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

    std::cout << shuffled.size();
   
    return EXIT_SUCCESS;
}