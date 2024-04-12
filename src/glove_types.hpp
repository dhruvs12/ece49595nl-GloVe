#pragma once
#include <cstdint>

using idx_t = uint32_t;
using fp_t = double;

using cooccur_key_t = std::pair<idx_t, idx_t>;
using cooccur_value_t = fp_t;
using cooccur_map_iter_t = std::pair<cooccur_key_t, cooccur_value_t>;

typedef struct _cooccur_t {
    idx_t token1;
    idx_t token2;
    cooccur_value_t val;
} cooccur_t;