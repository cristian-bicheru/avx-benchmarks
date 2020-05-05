#pragma once

template<typename ret, typename... Args>
double time_func(ret func, Args&&... args);

template<typename ret, typename... Args>
double mean_time_func(long int iterations, ret func, Args&&... args);