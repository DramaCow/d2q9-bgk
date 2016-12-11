#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

// SIZE value should be set by host program as a build argument
#ifndef SIZE
  #define SIZE 1024*1024
#endif

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct
{
  float f0[SIZE];
  float f1[SIZE];
  float f2[SIZE];
  float f3[SIZE];
  float f4[SIZE];
  float f5[SIZE];
  float f6[SIZE];
  float f7[SIZE];
  float f8[SIZE];
} t_speed_soa;

void reduce(local  float*, global float*);

// =======================
// === ACCELERATE FLOW ===
// =======================

kernel void accelerate_flow_1(global t_speed* cells,
                              global int* obstacles,
                              int nx, int ny,
                              float density, float accel)
{
  // compute weighting factors
  const float w1a = density * accel / 9.0f;
  const float w2a = density * accel / 36.0f;
  // rows used for accelerating flow
  const int row2 = ny - 2;

  // get column index
  int jj = get_global_id(0);

  // if the cell is not occupied and
  // we don't send a negative density
  if (!obstacles[row2 * nx + jj]
      && (cells[row2 * nx + jj].speeds[3] - w1a) > 0.0f
      && (cells[row2 * nx + jj].speeds[6] - w2a) > 0.0f
      && (cells[row2 * nx + jj].speeds[7] - w2a) > 0.0f)
  {
    // increase 'east-side' densities 
    cells[row2 * nx + jj].speeds[1] += w1a;
    cells[row2 * nx + jj].speeds[5] += w2a;
    cells[row2 * nx + jj].speeds[8] += w2a;
    // decrease 'west-side' densities 
    cells[row2 * nx + jj].speeds[3] -= w1a;
    cells[row2 * nx + jj].speeds[6] -= w2a;
    cells[row2 * nx + jj].speeds[7] -= w2a;
  }
}

kernel void accelerate_flow_2(global t_speed* cells,
                              global int* obstacles,
                              int nx, int ny,
                              float density, float accel)
{
  // compute weighting factors
  const float w1a = density * accel / 9.0f;
  const float w2a = density * accel / 36.0f;
  // rows used for accelerating flow
  const int row1 = ny - 1;
  const int row2 = ny - 2;
  const int row3 = ny - 3;

  // get column index
  int jj = get_global_id(0);

  int x_e = (jj == nx - 1) ? (0) : (jj + 1);
  int x_w = (jj == 0) ? (nx - 1) : (jj - 1);

  // if the cell is not occupied and
  // we don't send a negative density
  if (!obstacles[row2 * nx + jj]
      && (cells[row2 * nx + x_w].speeds[1] - w1a) > 0.0f
      && (cells[row1 * nx + x_w].speeds[8] - w2a) > 0.0f
      && (cells[row3 * nx + x_w].speeds[5] - w2a) > 0.0f)
  {
    // increase 'east-side' densities 
    cells[row2 * nx + x_e].speeds[3] += w1a;
    cells[row1 * nx + x_e].speeds[7] += w2a;
    cells[row3 * nx + x_e].speeds[6] += w2a;
    // decrease 'west-side' densities 
    cells[row2 * nx + x_w].speeds[1] -= w1a;
    cells[row1 * nx + x_w].speeds[8] -= w2a;
    cells[row3 * nx + x_w].speeds[5] -= w2a;
  }
}

// =========================
// === PROPAGATE-COLLIDE ===
// =========================

kernel void propagate_collide_1(global t_speed* cells,
                                global int* obstacles,
                                int nx, int ny, float omega,
                                local float* local_sums,
                                global float* partial_sums)
{
  int ii = get_global_id(1);
  int jj = get_global_id(0);

  int li = get_local_id(1);
  int lj = get_local_id(0);

  int nlx = get_local_size(0);

  // collision constants
  const float w[NSPEEDS] = { 4.0f / 9.0f, 
                             1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 
                             1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };
  const int u[NSPEEDS][2] = { {  0,  0 }, {  1,  0 }, {  0,  1 },
                              { -1,  0 }, {  0, -1 }, {  1,  1 },
                              { -1,  1 }, { -1, -1 }, {  1, -1 } };

  float tot_u = 0.0f;

  if (!obstacles[ii * nx + jj])
  { 
    // === PROPAGATE === aka. streaming
    // determine indices of axis-direction neighbours
    // respecting periodic boundary conditions (wrap around)
    int y_n = (ii == ny - 1) ? (0) : (ii + 1);
    //int y_n = max(ny - 1, 0);
    int x_e = (jj == nx - 1) ? (0) : (jj + 1);
    int y_s = (ii == 0) ? (ny - 1) : (ii - 1);
    int x_w = (jj == 0) ? (nx - 1) : (jj - 1);

    // propagate densities to neighbouring cells, following
    // appropriate directions of travel and writing into
    // scratch space grid
    float tmp_speeds[NSPEEDS];
    tmp_speeds[0] = cells[ii  * nx + jj ].speeds[0]; // central cell, no movement
    tmp_speeds[1] = cells[ii  * nx + x_w].speeds[1]; // west 
    tmp_speeds[2] = cells[y_s * nx + jj ].speeds[2]; // south
    tmp_speeds[3] = cells[ii  * nx + x_e].speeds[3]; // east
    tmp_speeds[4] = cells[y_n * nx + jj ].speeds[4]; // north
    tmp_speeds[5] = cells[y_s * nx + x_w].speeds[5]; // south-west
    tmp_speeds[6] = cells[y_s * nx + x_e].speeds[6]; // south-east
    tmp_speeds[7] = cells[y_n * nx + x_e].speeds[7]; // north-east
    tmp_speeds[8] = cells[y_n * nx + x_w].speeds[8]; // north-west 

    // === COLLISION === don't consider occupied cells
    // compute local density total
    float local_density = 0.0f;
    for (int kk = 0; kk < NSPEEDS; ++kk)
    {
      local_density += tmp_speeds[kk];
    }

    // compute x velocity component
    float u_x = ( 
                  + tmp_speeds[1]
                  + tmp_speeds[5]
                  + tmp_speeds[8]
                  - tmp_speeds[3]
                  - tmp_speeds[6]
                  - tmp_speeds[7] 
                ) / local_density;
                 
    // compute y velocity component
    float u_y = ( 
                  + tmp_speeds[2]
                  + tmp_speeds[5]
                  + tmp_speeds[6]
                  - tmp_speeds[4]
                  - tmp_speeds[7]
                  - tmp_speeds[8] 
                ) / local_density;

    // velocity squared
    float u_sq = u_x * u_x + u_y * u_y;

    // equilibrium densities
    float omega_d_equ[NSPEEDS];
    for (int kk = 0; kk < NSPEEDS; ++kk) {
      // directional velocity components
      float u_kk = u[kk][0]*u_x + u[kk][1]*u_y;
      omega_d_equ[kk] = w[kk] * omega * local_density * (1.0f + 3.0f*u_kk + 4.5f*u_kk*u_kk - 1.5f*u_sq);
    }

    // relaxation step
    // store cells speeds in adjacent cells
    for (int kk = 0; kk < NSPEEDS; ++kk) {
      tmp_speeds[kk] *= (1.0f - omega);
      tmp_speeds[kk] += omega_d_equ[kk];
    }
    cells[ii  * nx + jj ].speeds[0] = tmp_speeds[0]; // central cell, no movement
    cells[ii  * nx + x_w].speeds[1] = tmp_speeds[3]; // west
    cells[y_s * nx + jj ].speeds[2] = tmp_speeds[4]; // south
    cells[ii  * nx + x_e].speeds[3] = tmp_speeds[1]; // east
    cells[y_n * nx + jj ].speeds[4] = tmp_speeds[2]; // north
    cells[y_s * nx + x_w].speeds[5] = tmp_speeds[7]; // south-west
    cells[y_s * nx + x_e].speeds[6] = tmp_speeds[8]; // south-east
    cells[y_n * nx + x_e].speeds[7] = tmp_speeds[5]; // north-east
    cells[y_n * nx + x_w].speeds[8] = tmp_speeds[6]; // north-west

    // accumulate the norm of x- and y- velocity components
    tot_u = sqrt(u_x * u_x + u_y * u_y);
  }

  // reduction
  local_sums[li * nlx + lj] = tot_u;
  barrier(CLK_LOCAL_MEM_FENCE);

  reduce(local_sums, partial_sums);
}

kernel void propagate_collide_2(global t_speed* cells,
                                global int* obstacles,
                                int nx, int ny, float omega,
                                local float* local_sums,
                                global float* partial_sums)
{
  int ii = get_global_id(1);
  int jj = get_global_id(0);

  int li = get_local_id(1);
  int lj = get_local_id(0);

  int nlx = get_local_size(0);

  // collision constants
  const float w[NSPEEDS] = { 4.0f / 9.0f, 
                             1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 
                             1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };
  const int u[NSPEEDS][2] = { {  0,  0 }, {  1,  0 }, {  0,  1 },
                              { -1,  0 }, {  0, -1 }, {  1,  1 },
                              { -1,  1 }, { -1, -1 }, {  1, -1 } };

  float tot_u = 0.0f;

  float nonblocked = !obstacles[ii * nx + jj] ? 1.0f : 0.0f;

  if (!obstacles[ii * nx + jj])
  { 
    // === "PROPAGATE" === aka. streaming
    // this iteration only uses local speeds (no need to look at neighbours)
    // [[note: there speeds are "facing" inwards]]
    float tmp_speeds[NSPEEDS];
    tmp_speeds[0] = cells[ii * nx + jj].speeds[0];
    tmp_speeds[1] = cells[ii * nx + jj].speeds[3];
    tmp_speeds[2] = cells[ii * nx + jj].speeds[4];
    tmp_speeds[3] = cells[ii * nx + jj].speeds[1];
    tmp_speeds[4] = cells[ii * nx + jj].speeds[2];
    tmp_speeds[5] = cells[ii * nx + jj].speeds[7];
    tmp_speeds[6] = cells[ii * nx + jj].speeds[8];
    tmp_speeds[7] = cells[ii * nx + jj].speeds[5];
    tmp_speeds[8] = cells[ii * nx + jj].speeds[6];

    // === COLLISION === don't consider occupied cells
    // compute local density total
    float local_density = 0.0f;
    for (int kk = 0; kk < NSPEEDS; ++kk)
    {
      local_density += tmp_speeds[kk];
    }

    // compute x velocity component
    float u_x = ( 
                   + tmp_speeds[1]
                   + tmp_speeds[5]
                   + tmp_speeds[8]
                   - tmp_speeds[3]
                   - tmp_speeds[6]
                   - tmp_speeds[7] 
                 ) / local_density;
                 
    // compute y velocity component
    float u_y = ( 
                   + tmp_speeds[2]
                   + tmp_speeds[5]
                   + tmp_speeds[6]
                   - tmp_speeds[4]
                   - tmp_speeds[7]
                   - tmp_speeds[8] 
                 ) / local_density;

    // velocity squared
    float u_sq = u_x * u_x + u_y * u_y;

    // equilibrium densities
    float d_equ[NSPEEDS];
    for (int kk = 0; kk < NSPEEDS; ++kk) {
      // directional velocity components
      float u_kk = u[kk][0]*u_x + u[kk][1]*u_y;
      d_equ[kk] = w[kk] * local_density * (1.0f + 3.0f*u_kk + 4.5f*u_kk*u_kk - 1.5f*u_sq);
    }

    // relaxation step
    // store cells speeds in current cell only
    cells[ii * nx + jj].speeds[0] = tmp_speeds[0] - omega*(tmp_speeds[0] - d_equ[0]);
    cells[ii * nx + jj].speeds[1] = tmp_speeds[1] - omega*(tmp_speeds[1] - d_equ[1]);
    cells[ii * nx + jj].speeds[2] = tmp_speeds[2] - omega*(tmp_speeds[2] - d_equ[2]);
    cells[ii * nx + jj].speeds[3] = tmp_speeds[3] - omega*(tmp_speeds[3] - d_equ[3]);
    cells[ii * nx + jj].speeds[4] = tmp_speeds[4] - omega*(tmp_speeds[4] - d_equ[4]);
    cells[ii * nx + jj].speeds[5] = tmp_speeds[5] - omega*(tmp_speeds[5] - d_equ[5]);
    cells[ii * nx + jj].speeds[6] = tmp_speeds[6] - omega*(tmp_speeds[6] - d_equ[6]);
    cells[ii * nx + jj].speeds[7] = tmp_speeds[7] - omega*(tmp_speeds[7] - d_equ[7]);
    cells[ii * nx + jj].speeds[8] = tmp_speeds[8] - omega*(tmp_speeds[8] - d_equ[8]);

    // accumulate the norm of x- and y- velocity components
    tot_u += sqrt(u_x * u_x + u_y * u_y);
  }

  // reduction
  local_sums[li * nlx + lj] = tot_u;
  barrier(CLK_LOCAL_MEM_FENCE);

  reduce(local_sums, partial_sums);
}

// =====================================
// === REDUCTION ON AVERAGE VELOCITY ===
// =====================================

void reduce(local  float* local_sums,                          
            global float* partial_sums)                        
{                                                          
  int num_wrk_items  = get_local_size(0) * get_local_size(1);                 
  //int local_id       = get_local_id(0) * get_local_size(0) + get_local_id(1);                   
  int local_id       = get_local_id(1) * get_local_size(0) + get_local_id(0);                   
  int group_id       = get_group_id(1) * get_num_groups(0) + get_group_id(0);                   

/*
  // single workgroup reduction
  float sum;                              
  int i;                                      

  if (local_id == 0) {                      
    sum = 0.0f;                            
   
    for (i=0; i < num_wrk_items; i++) {        
      sum += local_sums[i];             
    }                                     
   
    partial_sums[group_id] = sum;         
  }
*/
  
/*
  // interleaved addressing 
  for (int s = 1; s < num_wrk_items; s *= 2) {
    int idx = 2 * s * local_id;

    if (idx < num_wrk_items) {
      local_sums[idx] += local_sums[idx + s];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {                      
    partial_sums[group_id] = local_sums[0];
  }
*/

  // sequential addressing
  for (int s = num_wrk_items / 2; s > 0; s >>= 1) {
    if (local_id < s) {
      local_sums[local_id] += local_sums[local_id + s];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {                      
    partial_sums[group_id] = local_sums[0];
  }
}

void reduce2(global float* local_sums,                          
             global float* partial_sums)                        
{                                                          
  int num_wrk_items  = get_local_size(0) * get_local_size(1);                 
  int local_id       = get_local_id(0) * get_local_size(0) + get_local_id(1);                   
  int group_id       = get_group_id(1) * get_num_groups(0) + get_group_id(0);
  int num_wrk_groups = get_num_groups(0) * get_num_groups(1);
  int global_id      = get_global_id(1) * get_global_size(0) + get_global_id(0);

  if (group_id < num_wrk_groups) {
    for (int s = num_wrk_groups / 2; s > 0; s >>= 1) {
      
    }
  }
}
