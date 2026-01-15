#pragma once
#include "moke/runtime/common.hpp"
#include <type_traits>

namespace moke {
template <class... Ts> struct TypeList {
    constexpr static int size = sizeof...(Ts);
};

template <int N>
struct Int : std::integral_constant<int, N> {};

template <int... Ns>
using Ints = TypeList<Int<Ns>...>;

namespace traits {
    template <class T1, class T2>
    struct Concat;

    template <template <class...> class TypePack, class... T1, class... Ts>
    struct Concat<TypePack<T1...>, TypePack<Ts...>> {
        using type = TypePack<T1..., Ts...>;
    };
} // namespace traits

template <class T1, class T2>
using Concat = traits::Concat<T1, T2>::type;

namespace traits {
    template <class T, class List> struct Combine;

    template <template <class...> class TypePack, class T, class... Ts>
    struct Combine<T, TypePack<Ts...>> {
        using type = TypePack<TypePack<T, Ts>...>;
    };

    template <template <class...> class TypePack, class... Es, class... Ts>
    struct Combine<TypePack<Es...>, TypePack<Ts...>> {
        using type = TypePack<TypePack<Es..., Ts>...>;
    };
} // namespace traits

template <class T, class List>
using Combine = traits::Combine<T, List>::type;

namespace traits {
    template <class T, int Idx> struct Get;

    template <template <class...> class TypePack, class T, class... Ts, int Idx>
    struct Get<TypePack<T, Ts...>, Idx> {
        constexpr static int value = Get<TypePack<Ts...>, Idx - 1>::value;
        using type = typename Get<TypePack<Ts...>, Idx - 1>::type;
    };

    template <template <class...> class TypePack, class T, class... Ts>
    struct Get<TypePack<T, Ts...>, 0> {
        constexpr static int value = T::value;
        using type = T;
    };

    template <template <class...> class TypePack, int Idx>
    struct Get<TypePack<>, Idx> {
        static_assert(Idx >= 0, "Idx out of range");
    };
} // namespace traits

template <class T, int Idx>
using Get = traits::Get<T, Idx>;

template <class T, int Idx>
using GetType = Get<T, Idx>::type;

template <class T, int Idx>
constexpr auto GetValue = Get<T, Idx>::value;

namespace traits {
    template <class List1, class List2>
    struct ProductBinary;

    template <class... Lists>
    struct Product;

    template <class T, class... Ts, class List2> struct ProductBinary<TypeList<T, Ts...>, List2> {
        using Part1 = typename Combine<T, List2>::type;
        using Part2 = typename ProductBinary<TypeList<Ts...>, List2>::type;
        using type = typename Concat<Part1, Part2>::type;
    };

    template <class List2> struct ProductBinary<TypeList<>, List2> {
        using type = TypeList<>;
    };

    template <class List1, class List2> struct Product<List1, List2> {
        using type = typename ProductBinary<List1, List2>::type;
    };

    template <class List1, class List2, class... Lists>
    struct Product<List1, List2, Lists...> {
        using type = typename Product<typename Product<List1, List2>::type, Lists...>::type;
    };
} // namespace traits

template <class... Lists>
using Product = traits::Product<Lists...>::type;
} // namespace moke
