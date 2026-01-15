#pragma once
#include "moke/native.hpp"
#include "moke/runtime/memory.hpp"

namespace moke::fill {
template <class ValueType>
struct constant {
    ValueType value;

    constant(ValueType value = 0) : value{value} {}

    template <class T, class memory_space_t>
    void operator()(memory_space_t space, T *dest, size_t size) const {
        memory_set(space, dest, value, size);
    }

    template <class Container>
    decltype(auto) operator()(Container &&container) const {
        operator()(container.mem_space(), container.data(), container.size());
        return std::forward<Container>(container);
    }
};

class random {
private:
    float m_min, m_max;
    uint32_t m_seed{0};

public:
    random() : m_min{0.0}, m_max{1.0} {}

    random(float max) : m_min{0.0}, m_max{max} {}

    random(float min, float max) : m_min{min}, m_max{max} {}

    decltype(auto) seed(uint32_t seed) { return m_seed = seed, *this; }

    template <class memory_space_t, class T>
    void operator()(memory_space_t mem_space, T *dest, size_t length) const {
        return fill_random(mem_space, dest, length, m_min, m_max, m_seed);
    }

    template <class Container>
    decltype(auto) operator()(Container &&container) const {
        operator()(container.mem_space(), container.data(), container.size());
        return std::forward<Container>(container);
    }
};
} // namespace moke::fill
