
#include "cuda_runtime.h"
#include "KMeans_CUDA.hpp"
#include "cuda_check.cu"


__global__ void sum_and_counts(
    float *d_params, 
    float *d_centroids, 
    float *d_sums, 
    int *d_counts, 
    int n, 
    int d, 
    int k) {

    extern __shared__ float shared[];

    float *s_centroids = shared;
    float *s_sums = &s_centroids[k*d];
    float *s_counts = &s_sums[k];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if(tid < k*d) {
        s_centroids[tid] = d_centroids[tid];
    }
    if(tid < k) {
        s_counts[tid] = 0;
    }
    if(tid < k*d) {
        s_sums[tid] = d_sums[tid];
    }

    __syncthreads();

    if(idx < n) {
        const int idxd = d * idx;

        int best_class = -1;
        float dist;
        float min_dist = MAXFLOAT;

        for(int i = 0; i < k; ++i) {
            dist = 0;

            for(int j = 0; j < d; ++j) {
                dist += pow(d_params[j+idxd] - s_centroids[j+i*d], 2);
            }

            if(dist < min_dist) {
                min_dist = dist;
                best_class = i;
            }
        }

        atomicAdd(&s_counts[best_class], 1);
        
        for(int i = 0; i < d; ++i) {
            atomicAdd(&s_sums[i + best_class * d], d_params[i + idxd]);
        }

    }

    __syncthreads();

    if(tid < k*d) {
        atomicAdd(&d_sums[tid], d_sums[tid]);
    }
    if(tid < k) {
        atomicAdd(&d_counts[tid], (int)s_counts[tid]);
    }
}


__global__ void update_centroids(
    float *d_centroids,
    float *d_sums,
    int *d_counts,
    int d,
    int k,
    bool flag
) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    flag = false;

    if(idx < k) {

        int idxd = idx * d;
        for(int i = 0; i < d; ++i) {

            float new_centroid = d_sums[idxd + i] / d_counts[idx];

            if(new_centroid != d_centroids[idxd + i]) {
                flag = true;
                d_centroids[idxd + i] = new_centroid;
            }

        }
    }

}


//__global__ void calculate_error(int *d_errors, const int k, )


KMeans::KMeans(float *h_params, const int &&k, const unsigned h, const unsigned w) {

    this->h_ = h;
    this->w_ = w;
    this->k_ = std::move(k);

    // normalize data
    float *mins = new float[w];
    float *maxs = new float[w];

    // get mins and maxs of each columns
    for(unsigned i = 0; i < w * h; ++i) {
        mins[i % w] = min(mins[i%w], h_params[i]);
        maxs[i % w] = max(maxs[i%w], h_params[i]);
    }

    h_data_ = new float[w*h];
    h_errors_ = new int[k*k];

    for(unsigned i = 0; i < w*h; ++i) {
        float dif = maxs[i%w] - mins[i%w];

        if(dif == 0) {
            dif = maxs[i%w];
        }

        // process of MinMax normalizing
        h_data_[i] = (h_params[i] - mins[i%w]) / dif;
    }

    delete[] mins;
    delete[] maxs;

    h_centroids_ = new float[k*w];

    // generate centroids
    for(int i = 0; i < k; ++i) {
        int index = rand() % h; 

        for(int j = 0; j < w; ++j) {
            h_centroids_[i*w+j] = h_data_[index*w + j];
        }
    }

    CUDA_CHECK(cudaMalloc(&d_data_, w*h*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids_, k*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums_, k_ * w_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts_, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_errors_, k*k*sizeof(int)));


    CUDA_CHECK(cudaMemcpy(d_data_, h_data_, w*h*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids_, h_centroids_, k*sizeof(float), cudaMemcpyHostToDevice));
}

KMeans::~KMeans() {

    delete[] h_data_;
    delete[] h_centroids_;

    CUDA_CHECK(cudaFree(d_data_));
    CUDA_CHECK(cudaFree(d_centroids_));
    CUDA_CHECK(cudaFree(d_sums_));
    CUDA_CHECK(cudaFree(d_counts_));

}

void KMeans::fit(int iters) {
    
    CUDA_CHECK(cudaMemset(d_sums_, 0, w_ * k_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_counts_, 0, k_ * sizeof(int)));

    int threads_per_block = static_cast<int>(pow(2, ceil(log2(w_))));
    int blocks = (h_ + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = 2*k_*w_ + k_;

    bool h_flag = false;
    bool *d_flag;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(float)));

    for(int i = 0; i < iters; ++i) {

        sum_and_counts<<<blocks, threads_per_block, shared_mem_size>>>(d_data_, d_centroids_, d_sums_, d_counts_, h_, w_, k_);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int blocks1 = (k_ + threads_per_block - 1) / threads_per_block;

        update_centroids<<<blocks1, threads_per_block>>>(d_centroids_, d_sums_, d_counts_, w_, k_, d_flag);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));

        if(!h_flag) {
            break;
        }
    }


}



std::vector<int> KMeans::prediction() {

    std::vector<int> preds;

    for(int i = 0; i < h_; ++i) {

        int best_cluster = 0;
        float dist = 0;
        
        for(int j = 0; j < w_; ++j) {
            dist += pow(h_data_[i*w_ + j] - h_centroids_[j], 2);
        }
        float min_dist = dist;

        for(int c = 1; c < k_; ++c) {
            dist = 0;

            for(int j = 0; j < w_; ++j) {
                dist += pow(h_data_[j + i*w_] - h_centroids_[j + c*w_], 2);
            }

            if(dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }

        preds.push_back(best_cluster);
    }

    return preds;
}

void KMeans::matrix_error() {

}