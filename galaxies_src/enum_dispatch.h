/* A utility to perform dynamic dispatch on a function that has enum template parameters.
 *
 * Requirements:
 *    1) wrap the function in a templated struct. The struct template parameters need to
 *       be ints (convert internally).
 *
 *    2) the function needs to be a static member of that struct, with name "apply".
 *
 *    3) function signature *CANNOT* depend on template parameters.
 *
 *    4) the individual enums need an "end_" marker so we can figure out their length.
 *
 *    5) the function object needs to have a static constexpr bool method named "allowed",
 *       without arguments. This is convenient if there are certain combinations of template
 *       parameters which are illegal (otherwise one needs to compile stubs for all these
 *       combinations or the linker complains).
 *
 * TODO why is it impossible to convert ints into enums inside the dispatcher?
 *      Seems pretty strange...
 *
 */


#ifndef ENUM_DISPATCH_H
#define ENUM_DISPATCH_H

#include <array>
#include <functional>

#include "err.h"

template<template<int ...> typename F /* the function object wrapping the desired functionality */,
         typename ...Targs /* the enum types, in correct order */>
struct Dispatcher
{
    auto operator() (Targs ...args/* the dynamic enum values */) const
    // returns a function pointer that can be called with the dynamic arguments
    {
        auto indices = std::array {(int)args...};
        int idx = 0;
        for (int ii=0; ii<(int)sizeof...(Targs); ++ii)
        {
            CHECK(indices[ii] >= dims[ii], throw std::runtime_error("out of bounds!"));
            idx += indices[ii] * Nel(ii);
        }
        CHECK(!f_array[idx], throw std::runtime_error("combination of template parameters not allowed!"));
        return f_array[idx];
    }

private:
// {{{
    template<typename T>
    static constexpr int Ncases ()
    {
        if constexpr (std::is_same<T, bool>::value)
            return 2;
        else
            return (int)T::end_;
    }

    template<typename T1, typename ...T>
    struct helper
    {
        static constexpr auto Nel (int n=1)
        {
            if constexpr (sizeof...(T)==0)
                return n * Ncases<T1>();
            else
                return helper<T...>::Nel(Ncases<T1>() * n);
        }

        template<int i, int ...done>
        static constexpr auto from1d ()
        {
            if constexpr (sizeof...(T)==0)
            {
                constexpr auto out = F<done..., i>::apply;
                if constexpr (F<done..., i>::allowed())
                    return out;
                else
                    return static_cast<decltype(out)>(nullptr);
            }
            else
            {
                constexpr auto n = helper<T...>::Nel();
                return helper<T...>::template from1d<i%n, done..., i/n> ();
            }
        }

        template<int running=0, int ...done>
        static constexpr auto farr ()
        {
            constexpr auto n = helper<T1, T...>::Nel();
            if constexpr (sizeof...(done)==n)
                return std::array {helper<T1, T...>::template from1d<done>()...};
            else
                return helper<T1, T...>::template farr<running+1, done..., running>();
        }
    };

    static constexpr int dims[] = { Ncases<Targs>()... };
    static constexpr auto f_array = helper<Targs...>::template farr<>();

    auto Nel (int idx) const
    {
        int out = 1;
        for (int ii=idx+1; ii<(int)sizeof...(Targs); ++ii)
            out *= dims[ii];
        return out;
    }
// }}}
};
#endif
