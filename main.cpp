#include <iostream>
#include <cmath>
#include <immintrin.h>

#include "time_backend.cpp"

int add(int a, int b) {
    return a+b;
}

/**
void double_aligned_copy(const double* src, double* dest, int len) {
    asm volatile ("l%=:"
                  "vmovapd   (%%rbx), %%ymm0;"
                  "vmovapd   %%ymm0, (%%rcx);"
                  "addq      $32, %%rbx;"
                  "addq      $32, %%rcx;"
                  "dec       %%eax;"
                  "jnz       l%=;"
    :: "a" (len/4), "b" (src), "c" (dest)
    : "%ymm0");
}

void double_aligned_copy_unrolled(const double* src, double* dest, int len) {
    asm volatile ("l%=:"
                  "vmovapd   (%%rbx), %%ymm0;"
                  "vmovapd   32(%%rbx), %%ymm1;"
                  "vmovapd   64(%%rbx), %%ymm2;"
                  "vmovapd   96(%%rbx), %%ymm3;"
                  "vmovapd   128(%%rbx), %%ymm4;"
                  "vmovapd   160(%%rbx), %%ymm5;"
                  "vmovapd   192(%%rbx), %%ymm6;"
                  "vmovapd   224(%%rbx), %%ymm7;"
                  "vmovapd   %%ymm0, (%%rcx);"
                  "vmovapd   %%ymm1, 32(%%rcx);"
                  "vmovapd   %%ymm2, 64(%%rcx);"
                  "vmovapd   %%ymm3, 96(%%rcx);"
                  "vmovapd   %%ymm4, 128(%%rcx);"
                  "vmovapd   %%ymm5, 160(%%rcx);"
                  "vmovapd   %%ymm6, 192(%%rcx);"
                  "vmovapd   %%ymm7, 224(%%rcx);"
                  "addq      $256, %%rbx;"
                  "addq      $256, %%rcx;"
                  "dec       %%eax;"
                  "jnz       l%=;"
    :: "a" (len/32), "b" (src), "c" (dest)
    : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7");
}

void double_unaligned_copy(const double* src, double* dest, int len) {
    asm volatile ("l%=:"
                  "vmovupd   (%%rbx), %%ymm0;"
                  "vmovupd   %%ymm0, (%%rcx);"
                  "addq      $32, %%rbx;"
                  "addq      $32, %%rcx;"
                  "dec       %%eax;"
                  "jnz       l%=;"
    :: "a" (len/4), "b" (src), "c" (dest)
    : "%ymm0");
}**/

void intrin_double_aligned_copy(const double* src, double* dest, int len) {
    for (int i = 0; i < len-4; i+= 4) {
        _mm256_stream_pd(&dest[i], _mm256_load_pd(&src[i]));
    }
}

void intrin_double_unaligned_copy(const double* src, double* dest, int len) {
    for (int i = 0; i < len-4; i+= 4) {
        _mm256_storeu_pd(&dest[i], _mm256_loadu_pd(&src[i]));
    }
}

void intrin_float_aligned_copy(const float* src, float* dest, int len) {
    for (int i = 0; i < len-8; i+= 8) {
        _mm256_stream_ps(&dest[i], _mm256_load_ps(&src[i]));
    }
}

void intrin_float_unaligned_copy(const float* src, float* dest, int len) {
    for (int i = 0; i < len-8; i+= 8) {
        _mm256_storeu_ps(&dest[i], _mm256_loadu_ps(&src[i]));
    }
}

void intrin_float_aligned_fast_div(const float* src1, const float* src2, float* dest, int len) {
    for (int i = 0; i < len-8; i+= 8) {
        _mm256_stream_ps(&dest[i], _mm256_div_ps(_mm256_load_ps(&src1[i]), _mm256_load_ps(&src2[i])));
    }
}

void intrin_float_unaligned_divop(const float* src1, const float* src2, float* dest, int len) {
    for (int i = 0; i < len-8; i+= 8) {
        _mm256_storeu_ps(&dest[i], _mm256_mul_ps(_mm256_loadu_ps(&src1[i]), _mm256_rcp_ps(_mm256_loadu_ps(&src2[i]))));
    }
}

int main() {
    const int len = pow(10, 8);
    const double nsts = 1.e9;
    const double iters = 100;
    double* src;
    double* dest;
    float* srcf;
    float* src2f;
    float* destf;
    double dt;

    /** Double Testing **/
    src = static_cast<double *>(aligned_alloc(32, len *sizeof(double)));
    dest = static_cast<double *>(aligned_alloc(32, len *sizeof(double)));

    // warmup
    for (int i = 0; i < iters; i++) {
        intrin_double_aligned_copy(src, dest, len);
        intrin_double_unaligned_copy(src, dest, len);
    }

    dt = mean_time_func(iters, intrin_double_aligned_copy, src, dest, len);
    std::cout << "Aligned Store/Load of " << len << " doubles: " << dt/nsts << "s." << std::endl;
    std::cout << len*nsts/dt << " doubles/s" << std::endl;

    dt = mean_time_func(iters, intrin_double_unaligned_copy, src, dest, len);
    std::cout << "Aligned Store/Load With Unaligned Instruction of " << len << " doubles: " << dt/nsts << "s." << std::endl;
    std::cout << len*nsts/dt << " doubles/s" << std::endl;

    free(src);
    free(dest);
    src = static_cast<double *>(malloc(len *sizeof(double)));
    dest = static_cast<double *>(malloc(len *sizeof(double)));

    // warmup
    for (int i = 0; i < iters; i++) {
        intrin_double_unaligned_copy(src, dest, len);
    }

    dt = mean_time_func(iters, intrin_double_unaligned_copy, src, dest, len);
    std::cout << "Unaligned Store/Load of " << len << " doubles: " << dt/nsts << "s." << std::endl;
    std::cout << len*nsts/dt << " doubles/s" << std::endl;

    free(src);
    free(dest);
    ////



    /** Float Testing **/
    srcf = static_cast<float *>(aligned_alloc(32, len *sizeof(float)));
    destf = static_cast<float *>(aligned_alloc(32, len *sizeof(float)));

    // warmup
    for (int i = 0; i < iters; i++) {
        intrin_float_aligned_copy(srcf, destf, len);
        intrin_float_unaligned_copy(srcf, destf, len);
    }

    dt = mean_time_func(iters, intrin_float_aligned_copy, srcf, destf, len);
    std::cout << "Aligned Store/Load of " << len << " floats: " << dt/nsts << "s." << std::endl;
    std::cout << len*nsts/dt << " floats/s" << std::endl;

    dt = mean_time_func(iters, intrin_float_unaligned_copy, srcf, destf, len);
    std::cout << "Aligned Store/Load With Unaligned Instruction of " << len << " floats: " << dt/nsts << "s." << std::endl;
    std::cout << len*nsts/dt << " floats/s" << std::endl;

    free(srcf);
    free(destf);
    srcf = static_cast<float *>(malloc(len *sizeof(float)));
    destf = static_cast<float *>(malloc(len *sizeof(float)));

    // warmup
    for (int i = 0; i < iters; i++) {
        intrin_float_unaligned_copy(srcf, destf, len);
    }

    dt = mean_time_func(iters, intrin_float_unaligned_copy, srcf, destf, len);
    std::cout << "Unaligned Store/Load of " << len << " floats: " << dt/nsts << "s." << std::endl;
    std::cout << len*nsts/dt << " floats/s" << std::endl;

    free(srcf);
    free(destf);
    ////



    /** FLOP Testing **/
    srcf = static_cast<float *>(aligned_alloc(32, len *sizeof(float)));
    src2f = static_cast<float *>(aligned_alloc(32, len *sizeof(float)));
    destf = static_cast<float *>(aligned_alloc(32, len *sizeof(float)));
    dt = mean_time_func(iters, intrin_float_aligned_fast_div, srcf, src2f, destf, len);
    std::cout << "Divison With Intrinsic  " << len << " floats: " << dt/nsts << "s." << std::endl;
    std::cout << len*nsts/dt << " floats/s" << std::endl;

    dt = mean_time_func(iters, intrin_float_unaligned_divop, srcf, src2f, destf, len);
    std::cout << "Division With Reciprocal Intrinsic  " << len << " floats: " << dt/nsts << "s." << std::endl;
    std::cout << len*nsts/dt << " floats/s" << std::endl;

    return 0;
}
