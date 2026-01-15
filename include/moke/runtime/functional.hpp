#pragma once
#include "moke/runtime/functional/compare.hpp"
#include "moke/runtime/functional/fill.hpp"
#include "moke/runtime/vector.hpp"

namespace moke {
template <class Container, class Functor>
decltype(auto) operator|(Container &&container, Functor &&func) {
    return func(std::forward<Container>(container));
}
} // namespace moke
