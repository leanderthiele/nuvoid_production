#ifndef TIMING_H
#define TIMING_H

#include <chrono>

inline auto start_time ()
{
    return std::chrono::steady_clock::now();
}

inline float get_time (decltype(start_time()) start)
{
    auto end = std::chrono::steady_clock::now();
    return (std::chrono::duration<float> (end - start)).count();
}

#endif // TIMING_H
