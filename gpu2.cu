#include "common.h"
#include <cuda.h>
#include <math.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// Adjusted global variables
int thread_count, block_count;
class ParticleWrapper;
ParticleWrapper** host_particle_grid;
__constant__ ParticleWrapper** device_particle_grid;
ParticleWrapper* host_particle_wrappers;
__constant__ ParticleWrapper* device_particle_wrappers;
__constant__ particle_t* device_parts;
__constant__ int num_particles;
__constant__ double sim_size;
int host_grid_width;
__constant__ int grid_width;
double grid_resolution = cutoff * 1.0001;
__constant__ double device_grid_step;
int* host_grid_offsets;
__constant__ int* device_grid_offsets;



class ParticleWrapper {
public:
    particle_t* particle;
    int grid_idx;
    int index;
};

// Device function refactoring
__device__ inline void compute_force(particle_t &p, particle_t &neighbor) {
    double dx = neighbor.x - p.x;
    double dy = neighbor.y - p.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double coefficient = (1.0 - cutoff * rsqrt(r2)) / (r2 * mass);
    
    p.ax += coefficient * dx;
    p.ay += coefficient * dy;
}

// Functor for setting grid indices
class AssignGridIndex {
public:
    __device__ void operator()(ParticleWrapper& wrapper) {
        particle_t& p = *wrapper.particle;
        int gx = 1 + static_cast<int>(p.x / device_grid_step);
        int gy = 1 + static_cast<int>(p.y / device_grid_step);
        int grid_idx = gx + (grid_width + 2) * gy;
        
        wrapper.grid_idx = grid_idx;
        atomicAdd(device_grid_offsets + grid_idx, 1);
    }
};

// Functor for sorting into grid
class SortIntoGrid {
public:
    __device__ void operator()(ParticleWrapper& wrapper) {
        int g_idx = wrapper.grid_idx;
        int offset = atomicSub(device_grid_offsets + g_idx, 1);
        device_particle_grid[offset - 1] = &wrapper;
    }
};

// Sorting function refactor
void organize_particles(particle_t* host_parts, int num_particles, double sim_size) {
    thrust::for_each_n(thrust::device, thrust::device_ptr<ParticleWrapper>(host_particle_wrappers), num_particles, AssignGridIndex());
    thrust::inclusive_scan(thrust::device, thrust::device_ptr<int>(host_grid_offsets), thrust::device_ptr<int>(host_grid_offsets + (host_grid_width + 2) * (host_grid_width + 2)), thrust::device_ptr<int>(host_grid_offsets));
    thrust::for_each_n(thrust::device, thrust::device_ptr<ParticleWrapper>(host_particle_wrappers), num_particles, SortIntoGrid());
}

// Optimized loop-based force application
__device__ void process_neighboring_cells(particle_t& p, int grid_idx) {
    int offsets[9] = {-grid_width-3, -grid_width-2, -grid_width-1, -1, 0, 1, grid_width+1, grid_width+2, grid_width+3};
    
    for (int k = 0; k < 9; ++k) {
        int neighbor_idx = grid_idx + offsets[k];
        for (int i = device_grid_offsets[neighbor_idx], e = device_grid_offsets[neighbor_idx + 1]; i < e; ++i) {
            compute_force(p, *device_particle_grid[i]->particle);
        }
    }
}

// CUDA Kernel for force computation
__global__ void compute_forces_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    ParticleWrapper* wrapper = device_particle_grid[idx];
    particle_t& p = *wrapper->particle;
    process_neighboring_cells(p, wrapper->grid_idx);
}

// CUDA Kernel for particle movement
__global__ void move_particles() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    ParticleWrapper* wrapper = device_particle_grid[idx];
    particle_t& p = *wrapper->particle;

    // Velocity Verlet integration
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Reflecting boundary conditions
    if (p.x < 0 || p.x > sim_size) {
        p.x = p.x < 0 ? -p.x : 2 * sim_size - p.x;
        p.vx = -p.vx;
    }

    if (p.y < 0 || p.y > sim_size) {
        p.y = p.y < 0 ? -p.y : 2 * sim_size - p.y;
        p.vy = -p.vy;
    }
    
    p.ax = p.ay = 0;
}

// Functor for initializing particle wrappers
class InitializeWrappers {
public:
    __device__ void operator()(int i) {
        ParticleWrapper& wrapper = device_particle_wrappers[i];
        wrapper.particle = device_parts + i;
        wrapper.index = i;
    }
};

// Simulation initialization
void init_simulation(particle_t* host_parts, int num_particles, double sim_size) {
    thread_count = 32;
    block_count = (num_particles + thread_count - 1) / thread_count;
    host_grid_width = static_cast<int>(sim_size / grid_resolution) + 1;

    cudaMalloc(&host_particle_grid, num_particles * sizeof(ParticleWrapper*));
    cudaMalloc(&host_grid_offsets, (host_grid_width + 2) * (host_grid_width + 2) * sizeof(int));
    cudaMalloc(&host_particle_wrappers, num_particles * sizeof(ParticleWrapper));

    cudaMemcpyToSymbol(device_parts, &host_parts, sizeof(particle_t*));
    cudaMemcpyToSymbol(num_particles, &num_particles, sizeof(int));
    cudaMemcpyToSymbol(sim_size, &sim_size, sizeof(double));
    cudaMemcpyToSymbol(device_particle_grid, &host_particle_grid, sizeof(ParticleWrapper**));
    cudaMemcpyToSymbol(device_grid_offsets, &host_grid_offsets, sizeof(int*));
    cudaMemcpyToSymbol(device_particle_wrappers, &host_particle_wrappers, sizeof(ParticleWrapper*));
    cudaMemcpyToSymbol(grid_width, &host_grid_width, sizeof(int));
    cudaMemcpyToSymbol(device_grid_step, &grid_resolution, sizeof(double));

    thrust::counting_iterator<int> iter(0);
    thrust::for_each_n(thrust::device, iter, num_particles, InitializeWrappers());
}

// Main simulation step
void simulate_one_step(particle_t* host_parts, int num_particles, double sim_size) {
    cudaMemset(host_grid_offsets, 0, (host_grid_width + 2) * (host_grid_width + 2) * sizeof(int));
    organize_particles(host_parts, num_particles, sim_size);
    compute_forces_kernel<<<block_count, thread_count>>>();
    move_particles<<<block_count, thread_count>>>();
}

