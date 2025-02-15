#ifndef KMEANS_CUDA_H
#define KMEANS_CUDA_N


#include <vector>

class KMeans {

public:
    KMeans(float *h_params, const int &&k, const unsigned h, const unsigned w);
    ~KMeans();

    void fit(int iters);

    std::vector<int> prediction();

    void matrix_error();


private:
    unsigned h_, w_; // size of table
    int k_;

    float *h_data_;
    float *h_centroids_;

    float *d_data_;
    float *d_centroids_;

    float *d_sums_;
    int *d_counts_;

    int *h_errors_;
    int *d_errors_;

};


#endif