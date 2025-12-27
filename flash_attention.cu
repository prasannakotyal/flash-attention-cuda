/**
 * Flash Attention - Memory-Efficient Exact Attention
 * 
 * Implementation based on:
 * "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * Tri Dao et al., 2022
 * 
 * This implementation demonstrates:
 * 1. Tiled computation to fit in SRAM (shared memory)
 * 2. Online softmax - incremental computation without full materialization
 * 3. O(N) memory complexity instead of O(N^2)
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iomanip>

// ============================================================================
// ANSI Color Codes
// ============================================================================
namespace Color {
    const char* RESET  = "\033[0m";
    const char* BOLD   = "\033[1m";
    const char* DIM    = "\033[2m";
    const char* RED    = "\033[31m";
    const char* GREEN  = "\033[32m";
    const char* YELLOW = "\033[33m";
    const char* CYAN   = "\033[36m";
    const char* WHITE  = "\033[37m";
}

// ============================================================================
// Configuration
// ============================================================================
constexpr int SEQ_LEN   = 512;    // Sequence length (N)
constexpr int HEAD_DIM  = 64;     // Head dimension (d)
constexpr int NUM_HEADS = 8;      // Number of attention heads
constexpr int BLOCK_M   = 32;     // Tile size for queries
constexpr int BLOCK_N = 32;       // Tile size for keys/values

constexpr float SOFTMAX_SCALE = 1.0f / 8.0f;  // 1/sqrt(64)

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << Color::RED << "[CUDA ERROR] "                         \
                      << cudaGetErrorString(err) << " at " << __FILE__         \
                      << ":" << __LINE__ << Color::RESET << std::endl;         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// Flash Attention Kernel
// 
// Algorithm:
// For each query block Q[i:i+Bm]:
//   Initialize: O = 0, l = 0, m = -inf
//   For each key/value block K[j:j+Bn], V[j:j+Bn]:
//     1. Load Q, K, V tiles into shared memory
//     2. Compute S = Q @ K^T * scale
//     3. Update running max: m_new = max(m, rowmax(S))
//     4. Compute P = exp(S - m_new)
//     5. Update sum: l_new = exp(m - m_new) * l + rowsum(P)
//     6. Update output: O = exp(m - m_new) * O + P @ V
//     7. m = m_new, l = l_new
//   Final: O = O / l
// ============================================================================
__global__
void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int d,
    float scale
) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int q_start = bx * BLOCK_M;
    
    if (q_start >= N) return;
    
    extern __shared__ float smem[];
    float* s_Q = smem;
    float* s_K = s_Q + BLOCK_M * HEAD_DIM;
    float* s_V = s_K + BLOCK_N * HEAD_DIM;
    float* s_S = s_V + BLOCK_N * HEAD_DIM;
    
    int q_row = q_start + tx;
    bool valid_q = (q_row < N);
    
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float out[HEAD_DIM];
    
    for (int i = 0; i < HEAD_DIM; i++) {
        out[i] = 0.0f;
    }
    
    // Load Q tile
    if (valid_q) {
        for (int i = 0; i < HEAD_DIM; i++) {
            s_Q[tx * HEAD_DIM + i] = Q[q_row * d + i];
        }
    } else {
        for (int i = 0; i < HEAD_DIM; i++) {
            s_Q[tx * HEAD_DIM + i] = 0.0f;
        }
    }
    
    int num_kv_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_row = kv_start + tx;
        bool valid_kv = (kv_row < N) && (tx < BLOCK_N);
        
        // Load K and V tiles
        if (valid_kv) {
            for (int i = 0; i < HEAD_DIM; i++) {
                s_K[tx * HEAD_DIM + i] = K[kv_row * d + i];
                s_V[tx * HEAD_DIM + i] = V[kv_row * d + i];
            }
        } else if (tx < BLOCK_N) {
            for (int i = 0; i < HEAD_DIM; i++) {
                s_K[tx * HEAD_DIM + i] = 0.0f;
                s_V[tx * HEAD_DIM + i] = 0.0f;
            }
        }
        
        __syncthreads();
        
        if (valid_q) {
            // Compute attention scores and find local max
            float local_max = -INFINITY;
            
            for (int j = 0; j < BLOCK_N; j++) {
                int kv_idx = kv_start + j;
                if (kv_idx >= N) continue;
                
                float score = 0.0f;
                for (int i = 0; i < HEAD_DIM; i++) {
                    score += s_Q[tx * HEAD_DIM + i] * s_K[j * HEAD_DIM + i];
                }
                score *= scale;
                
                s_S[tx * BLOCK_N + j] = score;
                local_max = fmaxf(local_max, score);
            }
            
            // Online softmax update
            float new_max = fmaxf(row_max, local_max);
            float correction = expf(row_max - new_max);
            float local_sum = 0.0f;
            
            for (int j = 0; j < BLOCK_N; j++) {
                int kv_idx = kv_start + j;
                if (kv_idx >= N) continue;
                
                float p = expf(s_S[tx * BLOCK_N + j] - new_max);
                s_S[tx * BLOCK_N + j] = p;
                local_sum += p;
            }
            
            float new_sum = correction * row_sum + local_sum;
            
            // Update output accumulator
            for (int i = 0; i < HEAD_DIM; i++) {
                out[i] *= correction;
            }
            
            for (int j = 0; j < BLOCK_N; j++) {
                int kv_idx = kv_start + j;
                if (kv_idx >= N) continue;
                
                float p = s_S[tx * BLOCK_N + j];
                for (int i = 0; i < HEAD_DIM; i++) {
                    out[i] += p * s_V[j * HEAD_DIM + i];
                }
            }
            
            row_max = new_max;
            row_sum = new_sum;
        }
        
        __syncthreads();
    }
    
    // Final normalization
    if (valid_q && row_sum > 0.0f) {
        for (int i = 0; i < HEAD_DIM; i++) {
            O[q_row * d + i] = out[i] / row_sum;
        }
    }
}

// ============================================================================
// Naive Attention Kernel (baseline for comparison)
// Materializes full N x N attention matrix
// ============================================================================
__global__
void naive_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ attn_matrix,
    int N,
    int d,
    float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    float max_val = -INFINITY;
    for (int col = 0; col < N; col++) {
        float score = 0.0f;
        for (int i = 0; i < d; i++) {
            score += Q[row * d + i] * K[col * d + i];
        }
        score *= scale;
        attn_matrix[row * N + col] = score;
        max_val = fmaxf(max_val, score);
    }
    
    float sum = 0.0f;
    for (int col = 0; col < N; col++) {
        float val = expf(attn_matrix[row * N + col] - max_val);
        attn_matrix[row * N + col] = val;
        sum += val;
    }
    
    for (int col = 0; col < N; col++) {
        attn_matrix[row * N + col] /= sum;
    }
    
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int col = 0; col < N; col++) {
            val += attn_matrix[row * N + col] * V[col * d + i];
        }
        O[row * d + i] = val;
    }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
void cpu_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d,
    float scale
) {
    float* attn = new float[N * N];
    
    for (int i = 0; i < N; i++) {
        float max_val = -INFINITY;
        for (int j = 0; j < N; j++) {
            float score = 0.0f;
            for (int k = 0; k < d; k++) {
                score += Q[i * d + k] * K[j * d + k];
            }
            score *= scale;
            attn[i * N + j] = score;
            max_val = std::fmax(max_val, score);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            attn[i * N + j] = std::exp(attn[i * N + j] - max_val);
            sum += attn[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            attn[i * N + j] /= sum;
        }
        
        for (int k = 0; k < d; k++) {
            float val = 0.0f;
            for (int j = 0; j < N; j++) {
                val += attn[i * N + j] * V[j * d + k];
            }
            O[i * d + k] = val;
        }
    }
    
    delete[] attn;
}

// ============================================================================
// Verification
// ============================================================================
float compute_max_error(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        max_err = std::fmax(max_err, std::fabs(a[i] - b[i]));
    }
    return max_err;
}

float compute_mean_error(const float* a, const float* b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += std::fabs(a[i] - b[i]);
    }
    return sum / size;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    // Memory sizes
    size_t matrix_bytes = SEQ_LEN * HEAD_DIM * sizeof(float);
    size_t attn_bytes = SEQ_LEN * SEQ_LEN * sizeof(float);
    size_t smem_size = (BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM + 
                        BLOCK_N * HEAD_DIM + BLOCK_M * BLOCK_N) * sizeof(float);
    
    // Host memory
    float *h_Q = new float[SEQ_LEN * HEAD_DIM];
    float *h_K = new float[SEQ_LEN * HEAD_DIM];
    float *h_V = new float[SEQ_LEN * HEAD_DIM];
    float *h_O_flash = new float[SEQ_LEN * HEAD_DIM];
    float *h_O_naive = new float[SEQ_LEN * HEAD_DIM];
    float *h_O_cpu = new float[SEQ_LEN * HEAD_DIM];
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) {
        h_Q[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_attn;
    CUDA_CHECK(cudaMalloc(&d_Q, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_attn, attn_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, matrix_bytes, cudaMemcpyHostToDevice));
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, SEQ_LEN, HEAD_DIM, SOFTMAX_SCALE);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // Naive GPU attention
    int naive_block = 256;
    int naive_grid = (SEQ_LEN + naive_block - 1) / naive_block;
    
    // Warmup
    naive_attention_kernel<<<naive_grid, naive_block>>>(
        d_Q, d_K, d_V, d_O, d_attn, SEQ_LEN, HEAD_DIM, SOFTMAX_SCALE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    naive_attention_kernel<<<naive_grid, naive_block>>>(
        d_Q, d_K, d_V, d_O, d_attn, SEQ_LEN, HEAD_DIM, SOFTMAX_SCALE);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    CUDA_CHECK(cudaMemcpy(h_O_naive, d_O, matrix_bytes, cudaMemcpyDeviceToHost));
    
    float naive_err = compute_max_error(h_O_naive, h_O_cpu, SEQ_LEN * HEAD_DIM);
    
    // Flash attention
    int flash_grid = (SEQ_LEN + BLOCK_M - 1) / BLOCK_M;
    int flash_block = BLOCK_M;
    
    // Warmup
    flash_attention_kernel<<<flash_grid, flash_block, smem_size>>>(
        d_Q, d_K, d_V, d_O, SEQ_LEN, HEAD_DIM, SOFTMAX_SCALE);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaEventRecord(start));
    flash_attention_kernel<<<flash_grid, flash_block, smem_size>>>(
        d_Q, d_K, d_V, d_O, SEQ_LEN, HEAD_DIM, SOFTMAX_SCALE);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float flash_time;
    CUDA_CHECK(cudaEventElapsedTime(&flash_time, start, stop));
    CUDA_CHECK(cudaMemcpy(h_O_flash, d_O, matrix_bytes, cudaMemcpyDeviceToHost));
    
    float flash_err = compute_max_error(h_O_flash, h_O_cpu, SEQ_LEN * HEAD_DIM);
    float flash_mean_err = compute_mean_error(h_O_flash, h_O_cpu, SEQ_LEN * HEAD_DIM);
    
    // Compute metrics
    float speedup_vs_cpu = cpu_time / flash_time;
    float speedup_vs_naive = naive_time / flash_time;
    float memory_reduction = (float)attn_bytes / smem_size;
    
    bool naive_passed = naive_err < 1e-4;
    bool flash_passed = flash_err < 1e-3;
    bool all_passed = naive_passed && flash_passed;
    
    // Print results
    std::cout << "\n";
    std::cout << Color::BOLD << "Flash Attention" << Color::RESET 
              << " - Memory-Efficient Exact Attention\n";
    std::cout << Color::DIM << "Tri Dao et al., 2022" << Color::RESET << "\n\n";
    
    std::cout << Color::CYAN << "[config]" << Color::RESET << "\n";
    std::cout << "  seq_len        : " << SEQ_LEN << "\n";
    std::cout << "  head_dim       : " << HEAD_DIM << "\n";
    std::cout << "  num_heads      : " << NUM_HEADS << "\n";
    std::cout << "  block_m        : " << BLOCK_M << "\n";
    std::cout << "  block_n        : " << BLOCK_N << "\n";
    std::cout << "  softmax_scale  : " << SOFTMAX_SCALE << "\n\n";
    
    std::cout << Color::CYAN << "[memory]" << Color::RESET << "\n";
    std::cout << "  Q, K, V, O     : " << std::fixed << std::setprecision(2) 
              << (matrix_bytes / 1024.0f) << " KB each\n";
    std::cout << "  naive (N*N)    : " << (attn_bytes / 1024.0f / 1024.0f) << " MB\n";
    std::cout << "  flash (tiles)  : " << (smem_size / 1024.0f) << " KB\n\n";
    
    std::cout << Color::CYAN << "[benchmark]" << Color::RESET << "\n";
    std::cout << std::left << std::setw(16) << "  method" 
              << std::right << std::setw(12) << "time (ms)" 
              << std::setw(12) << "speedup" 
              << std::setw(14) << "max_error" 
              << std::setw(10) << "status" << "\n";
    std::cout << "  " << std::string(60, '-') << "\n";
    
    // CPU row
    std::cout << std::left << std::setw(16) << "  cpu";
    std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(3) << cpu_time;
    std::cout << std::setw(12) << "1.0x";
    std::cout << std::setw(14) << "-";
    std::cout << std::setw(10) << "-" << "\n";
    
    // Naive GPU row
    std::cout << std::left << std::setw(16) << "  naive_gpu";
    std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(3) << naive_time;
    std::cout << std::setw(12) << std::setprecision(1) << (cpu_time / naive_time) << "x";
    std::cout << std::setw(14) << std::scientific << std::setprecision(2) << naive_err;
    if (naive_passed) {
        std::cout << Color::GREEN << std::setw(10) << "[PASS]" << Color::RESET;
    } else {
        std::cout << Color::RED << std::setw(10) << "[FAIL]" << Color::RESET;
    }
    std::cout << "\n";
    
    // Flash GPU row
    std::cout << std::left << std::setw(16) << "  flash_gpu";
    std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(3) << flash_time;
    std::cout << std::setw(12) << std::setprecision(1) << speedup_vs_cpu << "x";
    std::cout << std::setw(14) << std::scientific << std::setprecision(2) << flash_err;
    if (flash_passed) {
        std::cout << Color::GREEN << std::setw(10) << "[PASS]" << Color::RESET;
    } else {
        std::cout << Color::RED << std::setw(10) << "[FAIL]" << Color::RESET;
    }
    std::cout << "\n\n";
    
    std::cout << Color::CYAN << "[results]" << Color::RESET << "\n";
    std::cout << "  flash_vs_cpu   : " << std::fixed << std::setprecision(1) 
              << speedup_vs_cpu << "x faster\n";
    std::cout << "  flash_vs_naive : " << speedup_vs_naive << "x faster\n";
    std::cout << "  memory_saved   : " << memory_reduction << "x reduction\n\n";
    
    // Final status
    if (all_passed) {
        std::cout << Color::GREEN << "[PASS]" << Color::RESET 
                  << " All tests passed. Numerical verification successful.\n\n";
    } else {
        std::cout << Color::RED << "[FAIL]" << Color::RESET 
                  << " Verification failed. Check numerical precision.\n\n";
    }
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_attn));
    
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_flash;
    delete[] h_O_naive;
    delete[] h_O_cpu;
    
    return all_passed ? 0 : 1;
}
