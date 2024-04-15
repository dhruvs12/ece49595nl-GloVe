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

constexpr float beta1 = 0.9;
constexpr float beta2 = 0.999;
constexpr float epsilon = 1e-8;

static inline auto f = [](const fp_t &x) -> fp_t noexcept {
    return (x < x_max) 
        ? pow(x / x_max, alpha)
        : 1.;
};

int main() {
    std::vector<cooccur_t> X_ij (cooccur_size);
    std::vector<std::atomic<fp_t>> w_i (vocablen * vector_size), w_j (vocablen * vector_size);
    std::vector<std::atomic<fp_t>> b_i (vocablen), b_j (vocablen);
    std::vector<fp_t> m_w_i(vocablen * vector_size, 0.0), v_w_i(vocablen * vector_size, 0.0);
    std::vector<fp_t> m_w_j(vocablen * vector_size, 0.0), v_w_j(vocablen * vector_size, 0.0);
    std::vector<fp_t> m_b_i(vocablen, 0.0), v_b_i(vocablen, 0.0);
    std::vector<fp_t> m_b_j(vocablen, 0.0), v_b_j(vocablen, 0.0);

    std::array<std::string, vocablen> vocab;

    auto not_open = [](const auto &file, the auto &name) {
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

    // Initialize parameters
    std::random_device rd;
    std::mt19937_64 g(rd());
    std::uniform_real_distribution<fp_t> dist(-0.5 / vector_size, 0.5 / vector_size);

    for (auto &x : w_i) x.store(dist(g));
    for (auto &x : w_j) x.store(dist(g));
    for (auto &x : b_i) x.store(dist(g));
    for (auto &x : b_j) x.store(dist(g));

    std::atomic<fp_t> total_cost;
    for (idx_t epoch = 1; epoch <= iterations; epoch++) {
        total_cost.store(0);
        auto start = std::chrono::high_resolution_clock::now();

        std::for_each(
            std::execution::par,
            X_ij.begin(),
            X_ij.end(),
            [&](const cooccur_t &cc) {
                idx_t i_start = cc.token1 * vector_size, j_start = cc.token2 * vector_size;
                fp_t error = b_i[cc.token1].load(std::memory_order_relaxed) + b_j[cc.token2].load(std::memory_order_relaxed) - log(cc.val);

                for (idx_t k = 0; k < vector_size; k++) {
                    error += w_i[i_start + k].load(std::memory_order_relaxed) * w_j[j_start + k].load(std::memory_order_relaxed);
                }

                fp_t weight = f(cc.val);
                fp_t grad = error * weight * -2.0;
                fp_t grad_clip = std::clamp(grad, -gradient_clip, gradient_clip);

                for (idx_t k = 0; k < vector_size; k++) {
                    fp_t grad_w_i = grad_clip * w_j[j_start + k].load(std::memory_order_relaxed);
                    fp_t grad_w_j = grad_clip * w_i[i_start + k].load(std::memory_order_relaxed);

                    // Update m and v for w_i
                    m_w_i[i_start + k] = beta1 * m_w_i[i_start + k] + (1 - beta1) * grad_w_i;
                    v_w_i[i_start + k] = beta2 * v_w_i[i_start + k] + (1 - beta2) * (grad_w_i * grad_w_i);
                    fp_t m_hat_w_i = m_w_i[i_start + k] / (1 - pow(beta1, epoch));
                    fp_t v_hat_w_i = v_w_i[i_start + k] / (1 - pow(beta2, epoch));
                    w_i[i_start + k].fetch_add(-eta * m_hat_w_i / (sqrt(v_hat_w_i) + epsilon), std::memory_order_acq_rel);

                    // Update m and v for w_j
                    m_w_j[j_start + k] = beta1 * m_w_j[j_start + k] + (1 - beta1) * grad_w_j;
                    v_w_j[j_start + k] = beta2 * v_w_j[j_start + k] + (1 - beta2) * (grad_w_j * grad_w_j);
                    fp_t m_hat_w_j = m_w_j[j_start + k] / (1 - pow(beta1, epoch));
                    fp_t v_hat_w_j = v_w_j[j_start + k] / (1 - pow(beta2, epoch));
                    w_j[j_start + k].fetch_add(-eta * m_hat_w_j / (sqrt(v_hat_w_j) + epsilon), std::memory_order_acq_rel);
                }

                // Update m and v for b_i
                m_b_i[cc.token1] = beta1 * m_b_i[cc.token1] + (1 - beta1) * grad_clip;
                v_b_i[cc.token1] = beta2 * v_b_i[cc.token1] + (1 - beta2) * (grad_clip * grad_clip);
                fp_t m_hat_b_i = m_b_i[cc.token1] / (1 - pow(beta1, epoch));
                fp_t v_hat_b_i = v_b_i[cc.token1] / (1 - pow(beta2, epoch));
                b_i[cc.token1].fetch_add(-eta * m_hat_b_i / (sqrt(v_hat_b_i) + epsilon), std::memory_order_acq_rel);

                // Update m and v for b_j
                m_b_j[cc.token2] = beta1 * m_b_j[cc.token2] + (1 - beta1) * grad_clip;
                v_b_j[cc.token2] = beta2 * v_b_j[cc.token2] + (1 - beta2) * (grad_clip * grad_clip);
                fp_t m_hat_b_j = m_b_j[cc.token2] / (1 - pow(beta1, epoch));
                fp_t v_hat_b_j = v_b_j[cc.token2] / (1 - pow(beta2, epoch));
                b_j[cc.token2].fetch_add(-eta * m_hat_b_j / (sqrt(v_hat_b_j) + epsilon), std::memory_order_acq_rel);

                fp_t cost = error * error * weight;
                total_cost.fetch_add(cost, std::memory_order_acq_rel);
            }
        );

        auto stop = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        std::cout << std::format("Completed iteration {} of {}, took {} seconds. Total cost: {}\n", epoch, iterations, elapsed_time / 1000., total_cost.load() / X_ij.size());
    }

    // Output handling remains the same
}

