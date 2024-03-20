#include <iostream>
#include <cstdint>
#include <fstream>
#include <array>
#include <format>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include "filenames.hpp"
#include "glove_types.hpp"
#include "x_size.hpp"
#include "training_sizes.hpp"
#include "glove_const.hpp"

int main() {
    std::array<cooccur_t, cooccur_size> X_ij;
    std::array<std::array<fp_t, vocablen>, vector_size> w_i, w_j;
    std::array<fp_t, vocablen> b_i, b_j;

    std::array<std::string, vocablen> vocab;

    auto not_open = [](const auto &file, const auto &name) {
        if (!file.is_open()) {
            std::cerr << std::format("ERROR: {} not opened. Exiting...\n", name);
            return true;
        }
        return false;
    };

    std::ifstream coocurrence_in_file (cooccurrence_file);
    if (not_open(coocurrence_in_file, cooccurrence_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < cooccur_size; i++) {
        coocurrence_in_file.read(reinterpret_cast<char*>(&X_ij[i]), sizeof(cooccur_t));
    }

    coocurrence_in_file.close();

    std::ifstream vocab_in_file (vocab_file);
    if (not_open(vocab_in_file, vocab_file)) return EXIT_FAILURE;

    std::string word;
    idx_t i = 0, freq;
    while (vocab_in_file >> word >> freq) {
        vocab[i++] = word;
    }


    // Init params
    static std::random_device rd;
    static std::mt19937_64 g(rd());

    fp_t min = vocablen * vector_size * -2.; // total size = vocablen * (vector_size * 2 (w_ij) + 2 (b_ij))
    fp_t max = vocablen * vector_size * 2.;

    std::uniform_real_distribution<fp_t> dt (min, max);

    for (auto& row : w_i) std::generate(row.begin(), row.end(), [&]() { return dt(g); });
    for (auto& row : w_j) std::generate(row.begin(), row.end(), [&]() { return dt(g); });
    std::generate(b_i.begin(), b_i.end(), [&]() { return dt(g); });
    std::generate(b_j.begin(), b_j.end(), [&]() { return dt(g); });

    // train model

    

    std::ofstream vector_out_file (vector_file);
    if (not_open(vector_out_file, vector_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        vector_out_file << vocab[i] << " ";
        for (idx_t j = 0; j < vector_size; j++) {
            vector_out_file << w_i[i][j] << " ";
        }
        vector_out_file << std::endl;
    }

    vector_out_file.close();

    std::ofstream context_out_file (context_file);
    if (not_open(context_out_file, context_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        context_out_file << vocab[i] << " ";
        for (idx_t j = 0; j < vector_size; j++) {
            context_out_file << w_j[i][j] << " ";
        }
        context_out_file << std::endl;
    }

    context_out_file.close();

    std::ofstream bias_out_file (bias_file);
    if (not_open(bias_out_file, bias_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        bias_out_file << vocab[i] << " " << b_i[i] << " " << b_j[i] << std::endl;
    }

    bias_out_file.close();

    return EXIT_SUCCESS;
}
