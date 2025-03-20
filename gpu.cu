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

__global__ void assign_particles_to_grid(particle_t* particles, CellData* particle_cells, int* grid_offsets, int num_particles, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;  // Ensure we do not access out-of-bounds memory

    particle_t& p = particles[idx];

    // Compute the grid cell coordinates based on the particle's position
    int grid_x = static_cast<int>(p.x / device_grid_step);
    int grid_y = static_cast<int>(p.y / device_grid_step);
    int grid_idx = grid_y * grid_size + grid_x;

    // Ensure the index is within valid bounds
    if (grid_idx < 0 || grid_idx >= grid_size * grid_size) return;

    // Use atomicAdd to safely increment and get the correct position for this particle
    int pos = atomicAdd(&grid_offsets[grid_idx], 1);

    if (pos < num_particles) {
        particle_cells[pos].particle = &p;
        particle_cells[pos].grid_index = grid_idx;
        particle_cells[pos].particle_index = idx;
    }

    // Store the particle's information in the CellData structure
    particle_cells[pos].particle = &p;
    particle_cells[pos].grid_index = grid_idx;
    particle_cells[pos].particle_index = idx;
}

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
    // cudaMalloc(&host_grid_offsets, (host_grid_size + 2) * (host_grid_size + 2) * sizeof(int));
    cudaMalloc(&host_grid_offsets, (host_grid_size * host_grid_size) * sizeof(int));
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

    // Step 1: Assign particles to grid cells safely
    assign_particles_to_grid<<<(num_parts + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
        parts, host_particle_cells, host_grid_offsets, num_parts, host_grid_size
    );
    cudaDeviceSynchronize(); // Ensure all particles are correctly assigned before force computation

    // Step 2: Compute forces using the updated grid structure
    compute_forces_kernel<<<(num_parts + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>();
    cudaDeviceSynchronize(); // Ensure all forces are computed before updating positions

    // Step 3: Update positions of particles
    update_positions_kernel<<<(num_parts + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>();
    cudaDeviceSynchronize(); // Ensure all positions are updated before the next simulation step
}
