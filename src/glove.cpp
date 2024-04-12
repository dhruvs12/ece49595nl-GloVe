#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <execution>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
#include "filenames.hpp"
#include "glove_const.hpp"
#include "glove_types.hpp"
#include "training_sizes.hpp"
#include "x_size.hpp"

static inline auto f = [](const fp_t &x) -> fp_t noexcept {
    return (x < x_max) 
        ? pow(x / x_max, alpha)
        : 1.;
};

int main() {
    std::vector<cooccur_t> X_ij (cooccur_size);
    std::vector<std::atomic<fp_t>> w_i (vocablen * vector_size), w_j (vocablen * vector_size);
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

    std::uniform_real_distribution<fp_t> dt (-0.25 / vector_size, 0.25 / vector_size);
    
    for (auto &x : w_i) x.store(dt(g));
    for (auto &x : w_j) x.store(dt(g));
    for (auto &x : b_i) x.store(dt(g));
    for (auto &x : b_j) x.store(dt(g));

    // train model
    std::atomic<fp_t> total_cost;
    for (idx_t i = 1; i <= iterations; i++) {
        total_cost.store(0);
        auto start = std::chrono::high_resolution_clock::now();
        // if (i % 5) eta *= 0.95;
        if (i % 2) eta *= 1.5;
        if (i % 3) gradient_clip *= 1.3;

        std::for_each(
            std::execution::par,
            X_ij.begin(),
            X_ij.end(),
            [&] (const cooccur_t &cc) -> void {
                idx_t i_start = cc.token1 * vector_size, j_start = cc.token2 * vector_size;
                fp_t tot = b_i[cc.token1].load(std::memory_order_relaxed) + b_j[cc.token2].load(std::memory_order_relaxed) - log(cc.val);
                
                for (idx_t i = 0; i < vector_size; i++) 
                    tot += w_i[i_start + i].load(std::memory_order_relaxed) * w_j[j_start + i].load(std::memory_order_relaxed);
                
                fp_t grad = tot * f(cc.val) * -2;
                grad = std::clamp(grad, -gradient_clip, gradient_clip) * eta;

                for (idx_t i = 0; i < vector_size; i++) {
                    w_i[i_start + i].fetch_add(grad * w_j[j_start + i].load(std::memory_order_relaxed), std::memory_order_acq_rel);
                    w_j[j_start + i].fetch_add(grad * w_i[i_start + i].load(std::memory_order_relaxed), std::memory_order_acq_rel);
                }
                
                b_i[cc.token1].fetch_add(grad, std::memory_order_acq_rel);
                b_j[cc.token2].fetch_add(grad, std::memory_order_acq_rel);
                
                double cost = tot * tot * f(cc.val);
                total_cost.fetch_add(cost, std::memory_order_acq_rel);
            }
        );
        
        auto stop = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        std::cout << std::format("Completed iteration {} of {}, took {} seconds. Total cost: {}\n", i, iterations, elapsed_time / 1000., total_cost.load() / X_ij.size());
    } 
    
    std::ofstream vector_out_file (vector_file);
    if (not_open(vector_out_file, vector_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        vector_out_file << vocab[i] << " ";
        for (idx_t j = 0; j < vector_size; j++) {
            vector_out_file << w_i[(i * vector_size) + j].load(std::memory_order_relaxed) << " ";
        }
        vector_out_file << std::endl;
    }

    vector_out_file.close();

    std::ofstream context_out_file (context_file);
    if (not_open(context_out_file, context_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        context_out_file << vocab[i] << " ";
        for (idx_t j = 0; j < vector_size; j++) {
            context_out_file << w_j[(i * vector_size) + j].load(std::memory_order_relaxed) << " ";
        }
        context_out_file << std::endl;
    }

    context_out_file.close();

    std::ofstream bias_out_file (bias_file);
    if (not_open(bias_out_file, bias_file)) return EXIT_FAILURE;

    for (idx_t i = 0; i < vocablen; i++) {
        bias_out_file << vocab[i] << " " << b_i[i].load(std::memory_order_relaxed) << " " << b_j[i].load(std::memory_order_relaxed) << std::endl;
    }

    bias_out_file.close();

    return EXIT_SUCCESS;
}
