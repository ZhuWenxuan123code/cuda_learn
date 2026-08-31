#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do                                                                          \
    {                                                                           \
        const cudaError_t err = (call);                                         \
        if (err != cudaSuccess)                                                 \
        {                                                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << ": " << cudaGetErrorString(err) << "\n";            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int TILE_DIM = 16;
constexpr int SOFTMAX_THREADS = 256;
constexpr int WARMUP_ITERS = 10;
constexpr int BENCHMARK_ITERS = 100;

// Q, K: [N, d], row-major
// scores: [N, N], scores[row, col] = dot(Q[row], K[col]) * scale
__global__ void qk_transpose_kernel(const float *Q,
                                    const float *K,
                                    float *scores,
                                    int N,
                                    int d,
                                    float scale)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N)
        return;

    float dot = 0.0f;
    for (int k = 0; k < d; ++k)
    {
        dot += Q[row * d + k] * K[col * d + k];
    }
    scores[row * N + col] = dot * scale;
}

// One block computes one row of a numerically stable softmax.
// scores, probs: [N, N], row-major
__global__ void softmax_rows_kernel(const float *scores, float *probs, int N)
{
    __shared__ float reduction[SOFTMAX_THREADS];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= N)
        return;

    const float *score_row = scores + row * N;
    float *prob_row = probs + row * N;

    // Step 1: reduce the row maximum. Subtracting it before exp prevents
    // overflow for large positive scores.
    float local_max = -INFINITY;
    for (int col = tid; col < N; col += blockDim.x)
    {
        local_max = fmaxf(local_max, score_row[col]);
    }

    reduction[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduction[0];

    // Step 2: write unnormalized probabilities and reduce their sum.
    float local_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x)
    {
        const float value = expf(score_row[col] - row_max);
        prob_row[col] = value;
        local_sum += value;
    }

    reduction[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    const float row_sum = reduction[0];

    // Step 3: normalize the row.
    for (int col = tid; col < N; col += blockDim.x)
    {
        prob_row[col] /= row_sum;
    }
}

// probs: [N, N], V: [N, d], O: [N, d]
__global__ void pv_kernel(const float *probs,
                          const float *V,
                          float *O,
                          int N,
                          int d)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= d)
        return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
    {
        sum += probs[row * N + k] * V[k * d + col];
    }
    O[row * d + col] = sum;
}

void cpu_attention(const std::vector<float> &Q,
                   const std::vector<float> &K,
                   const std::vector<float> &V,
                   std::vector<float> &scores,
                   std::vector<float> &probs,
                   std::vector<float> &O,
                   int N,
                   int d)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    for (int row = 0; row < N; ++row)
    {
        float row_max = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < N; ++col)
        {
            float dot = 0.0f;
            for (int k = 0; k < d; ++k)
            {
                dot += Q[row * d + k] * K[col * d + k];
            }
            const float score = dot * scale;
            scores[row * N + col] = score;
            row_max = std::max(row_max, score);
        }

        float row_sum = 0.0f;
        for (int col = 0; col < N; ++col)
        {
            const float probability = std::exp(scores[row * N + col] - row_max);
            probs[row * N + col] = probability;
            row_sum += probability;
        }
        for (int col = 0; col < N; ++col)
        {
            probs[row * N + col] /= row_sum;
        }
    }

    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < d; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                sum += probs[row * N + k] * V[k * d + col];
            }
            O[row * d + col] = sum;
        }
    }
}

void initialize_inputs(std::vector<float> &Q,
                       std::vector<float> &K,
                       std::vector<float> &V)
{
    for (size_t i = 0; i < Q.size(); ++i)
    {
        // Three different deterministic sequences expose indexing and transpose
        // errors better than all-ones or constant data.
        Q[i] = 0.55f * std::sin(0.013f * static_cast<float>(i)) +
               0.05f * static_cast<float>(static_cast<int>(i % 7) - 3);
        K[i] = 0.45f * std::cos(0.017f * static_cast<float>(i + 11)) +
               0.04f * static_cast<float>(static_cast<int>(i % 5) - 2);
        V[i] = 0.35f * std::sin(0.019f * static_cast<float>(i + 23)) +
               0.03f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
}

struct ErrorStats
{
    float max_abs = 0.0f;
    float max_relative = 0.0f;
    size_t max_abs_index = 0;
};

ErrorStats compare_outputs(const std::vector<float> &actual,
                           const std::vector<float> &expected)
{
    ErrorStats stats;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const float abs_error = std::fabs(actual[i] - expected[i]);
        const float relative_error = abs_error / std::max(std::fabs(expected[i]), 1.0e-6f);
        if (abs_error > stats.max_abs)
        {
            stats.max_abs = abs_error;
            stats.max_abs_index = i;
        }
        stats.max_relative = std::max(stats.max_relative, relative_error);
    }
    return stats;
}

bool outputs_within_tolerance(const std::vector<float> &actual,
                              const std::vector<float> &expected,
                              float abs_tolerance,
                              float relative_tolerance)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const float abs_error = std::fabs(actual[i] - expected[i]);
        const float relative_error = abs_error / std::max(std::fabs(expected[i]), 1.0e-6f);
        // Relative error alone is not meaningful when the reference is near
        // zero. Accept an element when either tolerance is satisfied.
        if (abs_error > abs_tolerance && relative_error > relative_tolerance)
        {
            return false;
        }
    }
    return true;
}

float max_probability_row_sum_error(const std::vector<float> &probs, int N)
{
    float max_error = 0.0f;
    for (int row = 0; row < N; ++row)
    {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col)
        {
            sum += probs[row * N + col];
        }
        max_error = std::max(max_error, std::fabs(sum - 1.0f));
    }
    return max_error;
}

struct Timings
{
    float qk_ms = 0.0f;
    float softmax_ms = 0.0f;
    float pv_ms = 0.0f;
    float total_ms = 0.0f;
};

Timings benchmark_attention(const float *d_Q,
                             const float *d_K,
                             const float *d_V,
                             float *d_scores,
                             float *d_probs,
                             float *d_O,
                             int N,
                             int d)
{
    const dim3 block_2d(TILE_DIM, TILE_DIM);
    const dim3 qk_grid((N + TILE_DIM - 1) / TILE_DIM,
                       (N + TILE_DIM - 1) / TILE_DIM);
    const dim3 pv_grid((d + TILE_DIM - 1) / TILE_DIM,
                       (N + TILE_DIM - 1) / TILE_DIM);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    const auto launch_qk = [&]() {
        qk_transpose_kernel<<<qk_grid, block_2d>>>(d_Q, d_K, d_scores, N, d, scale);
    };
    const auto launch_softmax = [&]() {
        softmax_rows_kernel<<<N, SOFTMAX_THREADS>>>(d_scores, d_probs, N);
    };
    const auto launch_pv = [&]() {
        pv_kernel<<<pv_grid, block_2d>>>(d_probs, d_V, d_O, N, d);
    };

    for (int i = 0; i < WARMUP_ITERS; ++i)
    {
        launch_qk();
        launch_softmax();
        launch_pv();
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const auto measure = [&](const auto &launch) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < BENCHMARK_ITERS; ++i)
        {
            launch();
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaGetLastError());
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        return elapsed_ms / BENCHMARK_ITERS;
    };

    Timings timings;
    timings.qk_ms = measure(launch_qk);
    timings.softmax_ms = measure(launch_softmax);
    timings.pv_ms = measure(launch_pv);
    timings.total_ms = measure([&]() {
        launch_qk();
        launch_softmax();
        launch_pv();
    });

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return timings;
}

bool run_case(int N, int d, bool print_timing)
{
    std::cout << "\n=== Attention baseline: N=" << N << ", d=" << d << " ===\n";

    const size_t qkv_elements = static_cast<size_t>(N) * d;
    const size_t score_elements = static_cast<size_t>(N) * N;
    const size_t qkv_bytes = qkv_elements * sizeof(float);
    const size_t score_bytes = score_elements * sizeof(float);

    std::vector<float> h_Q(qkv_elements);
    std::vector<float> h_K(qkv_elements);
    std::vector<float> h_V(qkv_elements);
    std::vector<float> h_scores_ref(score_elements);
    std::vector<float> h_probs_ref(score_elements);
    std::vector<float> h_O_ref(qkv_elements);
    std::vector<float> h_probs(score_elements);
    std::vector<float> h_O(qkv_elements);

    initialize_inputs(h_Q, h_K, h_V);
    cpu_attention(h_Q, h_K, h_V, h_scores_ref, h_probs_ref, h_O_ref, N, d);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_scores = nullptr;
    float *d_probs = nullptr;
    float *d_O = nullptr;
    CHECK_CUDA(cudaMalloc(&d_Q, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_scores, score_bytes));
    CHECK_CUDA(cudaMalloc(&d_probs, score_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, qkv_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice));

    const dim3 block_2d(TILE_DIM, TILE_DIM);
    const dim3 qk_grid((N + TILE_DIM - 1) / TILE_DIM,
                       (N + TILE_DIM - 1) / TILE_DIM);
    const dim3 pv_grid((d + TILE_DIM - 1) / TILE_DIM,
                       (N + TILE_DIM - 1) / TILE_DIM);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    qk_transpose_kernel<<<qk_grid, block_2d>>>(d_Q, d_K, d_scores, N, d, scale);
    CHECK_CUDA(cudaGetLastError());
    softmax_rows_kernel<<<N, SOFTMAX_THREADS>>>(d_scores, d_probs, N);
    CHECK_CUDA(cudaGetLastError());
    pv_kernel<<<pv_grid, block_2d>>>(d_probs, d_V, d_O, N, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_probs.data(), d_probs, score_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, qkv_bytes, cudaMemcpyDeviceToHost));

    const ErrorStats output_error = compare_outputs(h_O, h_O_ref);
    const float probability_sum_error = max_probability_row_sum_error(h_probs, N);
    constexpr float kAbsTolerance = 1.0e-4f;
    constexpr float kRelativeTolerance = 1.0e-3f;
    constexpr float kProbabilitySumTolerance = 1.0e-4f;
    const bool correct = outputs_within_tolerance(h_O, h_O_ref, kAbsTolerance, kRelativeTolerance) &&
                         probability_sum_error <= kProbabilitySumTolerance;

    std::cout << std::scientific << std::setprecision(6)
              << "max abs error      : " << output_error.max_abs
              << " (O index " << output_error.max_abs_index << ")\n"
              << "max relative error : " << output_error.max_relative << "\n"
              << "max prob row-sum error: " << probability_sum_error << "\n"
              << (correct ? "verification: PASS\n" : "verification: FAIL\n");

    if (print_timing)
    {
        const Timings timings = benchmark_attention(d_Q, d_K, d_V, d_scores, d_probs, d_O, N, d);
        const double intermediate_mib =
            static_cast<double>(2 * score_bytes) / (1024.0 * 1024.0);
        std::cout << std::fixed << std::setprecision(4)
                  << "QK^T average       : " << timings.qk_ms << " ms\n"
                  << "softmax average    : " << timings.softmax_ms << " ms\n"
                  << "PV average         : " << timings.pv_ms << " ms\n"
                  << "end-to-end average : " << timings.total_ms << " ms\n"
                  << "scores + probs     : " << intermediate_mib << " MiB\n";
    }

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));
    CHECK_CUDA(cudaFree(d_O));
    return correct;
}

int main()
{
    const bool small_case_ok = run_case(17, 13, false);
    const bool default_case_ok = run_case(257, 64, true);
    return (small_case_ok && default_case_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
