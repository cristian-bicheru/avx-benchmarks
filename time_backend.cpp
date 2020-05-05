#include <chrono>
#include <utility>

#include "time_backend.h"

template<typename ret, typename... Args>
double time_func(ret func, Args&&... args) {
    auto t1 = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

template<typename ret, typename... Args>
double mean_time_func(long int iterations, ret func, Args&&... args) {
    double mean_time = 0;
    for (int i = 0; i < iterations; i++) {
        mean_time += time_func(func, args...);
    }
    return mean_time/iterations;
}