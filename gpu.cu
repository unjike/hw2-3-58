#include "common.h"
#include <cuda.h>
#include <math.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define NUM_THREADS 256

struct CellData;
__constant__ particle_t* device_particles;
__constant__ int total_particles;
__constant__ double sim_domain;
int host_grid_size;
__constant__ int device_grid_size;
double host_grid_step = cutoff * 1.00001;
__constant__ double device_grid_step;
CellData** host_grid;
__constant__ CellData** device_grid;
int* host_grid_offsets;
__constant__ int* device_grid_offsets;
CellData* host_particle_cells;
__constant__ CellData* device_particle_cells;

struct CellData {
    particle_t* particle;
    int grid_index;
    int particle_index;
};

__device__ void compute_interaction(particle_t &p1, particle_t &p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double force_coeff = (1 - cutoff * rsqrt(r2)) / r2 / mass;
    p1.ax += force_coeff * dx;
    p1.ay += force_coeff * dy;
}

__global__ void compute_forces_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_particles) return;
    CellData* cell = device_grid[idx];
    particle_t& p = *cell->particle;
    int grid_idx = cell->grid_index;
    for (int offset = -device_grid_size - 2; offset <= device_grid_size + 2; offset += device_grid_size + 2) {
        for (int j = -1; j <= 1; ++j) {
            int neighbor_idx = grid_idx + offset + j;
            for (int i = device_grid_offsets[neighbor_idx]; i < device_grid_offsets[neighbor_idx + 1]; ++i) {
                compute_interaction(p, *device_grid[i]->particle);
            }
        }
    }
}

__global__ void update_positions_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_particles) return;
    particle_t* p = device_grid[idx]->particle;
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    p->ax = p->ay = 0;
}

void init_simulation(particle_t* host_particles, int num_particles, double domain_size) {
    host_grid_size = static_cast<int>(domain_size / host_grid_step) + 1;
    cudaMalloc(&host_grid, num_particles * sizeof(CellData*));
    cudaMalloc(&host_grid_offsets, (host_grid_size + 2) * (host_grid_size + 2) * sizeof(int));
    cudaMalloc(&host_particle_cells, num_particles * sizeof(CellData));
    cudaMemcpyToSymbol(device_particles, &host_particles, sizeof(particle_t*));
    cudaMemcpyToSymbol(total_particles, &num_particles, sizeof(int));
    cudaMemcpyToSymbol(sim_domain, &domain_size, sizeof(double));
    cudaMemcpyToSymbol(device_grid, &host_grid, sizeof(CellData**));
    cudaMemcpyToSymbol(device_grid_offsets, &host_grid_offsets, sizeof(int*));
    cudaMemcpyToSymbol(device_particle_cells, &host_particle_cells, sizeof(CellData*));
    cudaMemcpyToSymbol(device_grid_size, &host_grid_size, sizeof(int));
    cudaMemcpyToSymbol(device_grid_step, &host_grid_step, sizeof(double));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    cudaMemset(host_grid_offsets, 0, (host_grid_size + 2) * (host_grid_size + 2) * sizeof(int));
    compute_forces_kernel<<<(num_parts + 31) / 32, 32>>>();
    update_positions_kernel<<<(num_parts + 31) / 32, 32>>>();
}
