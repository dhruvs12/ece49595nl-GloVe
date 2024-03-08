#pragma once
#include <cstdint>

using idx_t = uint32_t;
using cooccur_key_t = std::pair<idx_t, idx_t>;
using cooccur_value_t = long double;
using cooccur_map_iter_t = std::pair<cooccur_key_t, cooccur_value_t>;