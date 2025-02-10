
#include "cuda_runtime.h"

#include <iostream>
#include <cassert>

#define CUDA_CHECK(ans) {cudaAssert(ans, __FILE__, __LINE__);}

inline void cudaAssert(cudaError_t ans, const char *file, unsigned long line, bool is_abort=true) {
    
    if(ans != cudaSuccess) {

        std::cerr << "GPU error: " << cudaGetErrorString(ans) << ' ' << file << ' ' << line << '\n';

        assert(is_abort);

    }

}

