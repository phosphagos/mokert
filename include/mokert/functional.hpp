#pragma once
#include "mokert/functional/compare.hpp"
#include "mokert/functional/fill.hpp"
#include "mokert/vector.hpp"

namespace moke {
template <class Container, class Functor>
decltype(auto) operator|(Container &&container, Functor &&func) {
    return func(std::forward<Container>(container));
}
} // namespace moke
