#include <iostream>
#include <atomic>
#include <cstdint>
#include <execution>
#include <cmath>
#include <fstream>
#include <array>
#include <format>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include "filenames.hpp"
#include "glove_types.hpp"
#include "x_size.hpp"
#include "training_sizes.hpp"
#include "glove_const.hpp"


static inline auto f = [](const fp_t &x) -> fp_t {
    return (x < x_max) 
        ? powl(x / x_max, alpha)
        : 1.;
};

int main() {
    std::vector<cooccur_t> X_ij (cooccur_size);
    std::array<std::array<std::atomic<fp_t>, vector_size>, vocablen> w_i, w_j;
    //std::vector<std::vector<std::atomic<fp_t>>> w_i (vector_size, std::vector<std::atomic<fp_t>> (vocablen)), 
    //                                            w_j (vector_size, std::vector<std::atomic<fp_t>> (vocablen));
    std::vector<std::atomic<fp_t>> b_i (vocablen), b_j (vocablen);

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
        coocurrence_in_file.read(reinterpret_cast<char*>(&X_ij[i].token1), sizeof(X_ij[i].token1));
        coocurrence_in_file.read(reinterpret_cast<char*>(&X_ij[i].token2), sizeof(X_ij[i].token2));
        coocurrence_in_file.read(reinterpret_cast<char*>(&X_ij[i].val), sizeof(X_ij[i].val));
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

    std::uniform_real_distribution<fp_t> dt (-1 / vector_size, 1 / vector_size);

    for (auto& row : w_i) std::for_each(row.begin(), row.end(), [&](auto &x) { x.fetch_add(dt(g)); });
    for (auto& row : w_j) std::for_each(row.begin(), row.end(), [&](auto &x) { x.fetch_add(dt(g)); });
    std::for_each(b_i.begin(), b_i.end(), [&](auto &x) { x.fetch_add(dt(g)); });
    std::for_each(b_j.begin(), b_j.end(), [&](auto &x) { x.fetch_add(dt(g)); });


    // train model
    std::cout << "BEGIN\n";
    for (idx_t i = 1; i <= iterations; i++) {
        std::atomic<fp_t> total_cost = 0;
        auto start = std::chrono::high_resolution_clock::now();
        std::for_each(
            std::execution::seq,
            X_ij.begin(),
            X_ij.end(),
            [&] (const cooccur_t &cc) {
                fp_t tot = b_i[cc.token1].load() + b_j[cc.token2].load() - logl(cc.val);
                
                for (idx_t i = 0; i < vector_size; i++) 
                    tot += w_i[cc.token1][i].load() * w_j[cc.token2][i].load();
                
                fp_t grad = tot * f(cc.val) * -2 * eta;

                for (idx_t i = 0; i < vector_size; i++) {
                    w_i[cc.token1][i].fetch_add(grad * w_j[cc.token2][i].load());
                    w_j[cc.token2][i].fetch_add(grad * w_i[cc.token1][i].load());
                }
                
                b_i[cc.token1].fetch_add(grad);
                b_j[cc.token2].fetch_add(grad);

                total_cost.fetch_add(tot * tot * f(cc.val));
            }
        );
        
        auto stop = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();

        std::cout << std::format("Completed iteration {} of {}, took {} seconds. Total cost: {}\n", i, iterations, elapsed_time, total_cost.load());
    }
    
    std::ofstream vector_out_file (vector_file);
    if (not_open(vector_out_file, vector_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        vector_out_file << vocab[i] << " ";
        for (idx_t j = 0; j < vector_size; j++) {
            vector_out_file << w_i[i][j].load() << " ";
        }
        vector_out_file << std::endl;
    }

    vector_out_file.close();

    std::ofstream context_out_file (context_file);
    if (not_open(context_out_file, context_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        context_out_file << vocab[i] << " ";
        for (idx_t j = 0; j < vector_size; j++) {
            context_out_file << w_j[i][j].load() << " ";
        }
        context_out_file << std::endl;
    }

    context_out_file.close();

    std::ofstream bias_out_file (bias_file);
    if (not_open(bias_out_file, bias_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        bias_out_file << vocab[i] << " " << b_i[i].load() << " " << b_j[i].load() << std::endl;
    }

    bias_out_file.close();

    return EXIT_SUCCESS;
}
