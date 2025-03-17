#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

__constant__ particle_t* d_particles;
__constant__ int d_num_particles;
__constant__ double d_sim_size;
__constant__ int d_grid_width;
__constant__ double d_grid_step;

struct ParticleData {
    double x, y, vx, vy, ax, ay;
};

struct GridCell {
    int start_idx;
    int count;
};

__device__ inline void compute_interaction(particle_t &a, particle_t &b) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    double r2 = dx * dx + dy * dy;

    if (r2 >= cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double force_mag = (1.0 - cutoff / sqrt(r2)) / (r2 * mass);

    a.ax += force_mag * dx;
    a.ay += force_mag * dy;
}

struct AssignGrid {
    int grid_dim;
    double grid_step;
    int* grid_counts;

    AssignGrid(int grid_size, double step, int* counts)
        : grid_dim(grid_size), grid_step(step), grid_counts(counts) {}

    __device__ void operator()(particle_t& p) {
        int gx = min(max(int(p.x / grid_step), 0), grid_dim - 1);
        int gy = min(max(int(p.y / grid_step), 0), grid_dim - 1);
        int grid_idx = gy * grid_dim + gx;
        atomicAdd(&grid_counts[grid_idx], 1);
    }
};

void init_simulation(particle_t* h_particles, int num_particles, double sim_size) {
    int grid_size = int(sim_size / (cutoff * 1.1)) + 1;
    int total_cells = grid_size * grid_size;

    cudaMemcpyToSymbol(d_particles, &h_particles, sizeof(particle_t*));
    cudaMemcpyToSymbol(d_num_particles, &num_particles, sizeof(int));
    cudaMemcpyToSymbol(d_sim_size, &sim_size, sizeof(double));
    cudaMemcpyToSymbol(d_grid_width, &grid_size, sizeof(int));

    thrust::device_vector<int> grid_counts(total_cells, 0);
    thrust::for_each(thrust::device, h_particles, h_particles + num_particles, 
                     AssignGrid(grid_size, cutoff * 1.1, thrust::raw_pointer_cast(grid_counts.data())));

    thrust::inclusive_scan(thrust::device, grid_counts.begin(), grid_counts.end(), grid_counts.begin());
}

__global__ void compute_forces_gpu(particle_t* particles, int* grid_counts, int grid_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_num_particles) return;

    particle_t& p = particles[idx];
    int gx = min(max(int(p.x / d_grid_step), 0), grid_width - 1);
    int gy = min(max(int(p.y / d_grid_step), 0), grid_width - 1);
    int grid_idx = gy * grid_width + gx;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int neighbor_idx = grid_idx + dy * grid_width + dx;
            if (neighbor_idx < 0 || neighbor_idx >= grid_width * grid_width) continue;
            int start = (neighbor_idx == 0) ? 0 : grid_counts[neighbor_idx - 1];
            int end = grid_counts[neighbor_idx];

            for (int i = start; i < end; ++i) {
                compute_interaction(p, particles[i]);
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_num_particles) return;

    particle_t& p = particles[idx];

    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    if (p.x < 0 || p.x > d_sim_size) {
        p.x = (p.x < 0) ? -p.x : 2 * d_sim_size - p.x;
        p.vx *= -1;
    }
    if (p.y < 0 || p.y > d_sim_size) {
        p.y = (p.y < 0) ? -p.y : 2 * d_sim_size - p.y;
        p.vy *= -1;
    }

    p.ax = p.ay = 0;
}

void simulate_one_step(particle_t* h_particles, int num_particles, double sim_size) {
    int grid_size = int(sim_size / (cutoff * 1.1)) + 1;
    int total_cells = grid_size * grid_size;

    thrust::device_vector<int> grid_counts(total_cells, 0);
    thrust::for_each(thrust::device, h_particles, h_particles + num_particles, 
                     AssignGrid(grid_size, cutoff * 1.1, thrust::raw_pointer_cast(grid_counts.data())));
    thrust::inclusive_scan(thrust::device, grid_counts.begin(), grid_counts.end(), grid_counts.begin());

    int threads_per_block = 128;
    int num_blocks = (num_particles + threads_per_block - 1) / threads_per_block;
    
    compute_forces_gpu<<<num_blocks, threads_per_block>>>(h_particles, thrust::raw_pointer_cast(grid_counts.data()), grid_size);
    move_gpu<<<num_blocks, threads_per_block>>>(h_particles);
}
