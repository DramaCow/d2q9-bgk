#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

// SIZE value should be set by host program as a build argument
#ifndef SIZE
  #define SIZE 1024*1024
#endif

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
} t_speed;

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
      && (cells->f3[row2 * nx + jj] - w1a) > 0.0f
      && (cells->f6[row2 * nx + jj] - w2a) > 0.0f
      && (cells->f7[row2 * nx + jj] - w2a) > 0.0f)
  {
    // increase 'east-side' densities 
    cells->f1[row2 * nx + jj] += w1a;
    cells->f5[row2 * nx + jj] += w2a;
    cells->f8[row2 * nx + jj] += w2a;
    // decrease 'west-side' densities 
    cells->f3[row2 * nx + jj] -= w1a;
    cells->f6[row2 * nx + jj] -= w2a;
    cells->f7[row2 * nx + jj] -= w2a;
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
      && (cells->f1[row2 * nx + x_w] - w1a) > 0.0f
      && (cells->f8[row1 * nx + x_w] - w2a) > 0.0f
      && (cells->f5[row3 * nx + x_w] - w2a) > 0.0f)
  {
    // increase 'east-side' densities 
    cells->f3[row2 * nx + x_e] += w1a;
    cells->f7[row1 * nx + x_e] += w2a;
    cells->f6[row3 * nx + x_e] += w2a;
    // decrease 'west-side' densities 
    cells->f1[row2 * nx + x_w] -= w1a;
    cells->f8[row1 * nx + x_w] -= w2a;
    cells->f5[row3 * nx + x_w] -= w2a;
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
    tmp_speeds[0] = cells->f0[ii  * nx + jj ]; // central cell, no movement
    tmp_speeds[1] = cells->f1[ii  * nx + x_w]; // west 
    tmp_speeds[2] = cells->f2[y_s * nx + jj ]; // south
    tmp_speeds[3] = cells->f3[ii  * nx + x_e]; // east
    tmp_speeds[4] = cells->f4[y_n * nx + jj ]; // north
    tmp_speeds[5] = cells->f5[y_s * nx + x_w]; // south-west
    tmp_speeds[6] = cells->f6[y_s * nx + x_e]; // south-east
    tmp_speeds[7] = cells->f7[y_n * nx + x_e]; // north-east
    tmp_speeds[8] = cells->f8[y_n * nx + x_w]; // north-west 

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
    cells->f0[ii  * nx + jj ] = tmp_speeds[0]; // central cell, no movement
    cells->f1[ii  * nx + x_w] = tmp_speeds[3]; // west
    cells->f2[y_s * nx + jj ] = tmp_speeds[4]; // south
    cells->f3[ii  * nx + x_e] = tmp_speeds[1]; // east
    cells->f4[y_n * nx + jj ] = tmp_speeds[2]; // north
    cells->f5[y_s * nx + x_w] = tmp_speeds[7]; // south-west
    cells->f6[y_s * nx + x_e] = tmp_speeds[8]; // south-east
    cells->f7[y_n * nx + x_e] = tmp_speeds[5]; // north-east
    cells->f8[y_n * nx + x_w] = tmp_speeds[6]; // north-west

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

  //printf("%d, %d, %d, %d, %d\n", ii, jj, li, lj, nlx);

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
    tmp_speeds[0] = cells->f0[ii * nx + jj];
    tmp_speeds[1] = cells->f3[ii * nx + jj];
    tmp_speeds[2] = cells->f4[ii * nx + jj];
    tmp_speeds[3] = cells->f1[ii * nx + jj];
    tmp_speeds[4] = cells->f2[ii * nx + jj];
    tmp_speeds[5] = cells->f7[ii * nx + jj];
    tmp_speeds[6] = cells->f8[ii * nx + jj];
    tmp_speeds[7] = cells->f5[ii * nx + jj];
    tmp_speeds[8] = cells->f6[ii * nx + jj];

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
    cells->f0[ii * nx + jj] = tmp_speeds[0] - omega*(tmp_speeds[0] - d_equ[0]);
    cells->f1[ii * nx + jj] = tmp_speeds[1] - omega*(tmp_speeds[1] - d_equ[1]);
    cells->f2[ii * nx + jj] = tmp_speeds[2] - omega*(tmp_speeds[2] - d_equ[2]);
    cells->f3[ii * nx + jj] = tmp_speeds[3] - omega*(tmp_speeds[3] - d_equ[3]);
    cells->f4[ii * nx + jj] = tmp_speeds[4] - omega*(tmp_speeds[4] - d_equ[4]);
    cells->f5[ii * nx + jj] = tmp_speeds[5] - omega*(tmp_speeds[5] - d_equ[5]);
    cells->f6[ii * nx + jj] = tmp_speeds[6] - omega*(tmp_speeds[6] - d_equ[6]);
    cells->f7[ii * nx + jj] = tmp_speeds[7] - omega*(tmp_speeds[7] - d_equ[7]);
    cells->f8[ii * nx + jj] = tmp_speeds[8] - omega*(tmp_speeds[8] - d_equ[8]);

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
//*
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
//*/
}

kernel void reduce2(global float* partial_sums,  
                    local  float* local_sums,
                    global float* av_vels,
                    int num_wrk_groups,
                    int tt)                        
{                                                          
  int local_id       = get_local_id(0);
  int group_id       = get_group_id(0);
  int num_wrk_items  = get_local_size(0); 

  if (group_id == 0) { //&& local_id < num_wrk_items) {
    float sum = 0.0f;
    for (int i = local_id; i < num_wrk_groups; i += num_wrk_items) {
      sum += partial_sums[i];
    } 
    local_sums[local_id] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = num_wrk_items / 2; s > 0; s >>= 1) {
      if (local_id < s) {
        local_sums[local_id] += local_sums[local_id + s];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {                      
      av_vels[tt] = local_sums[0];
    }
  }
}

kernel void divide(global float* av_vels, float tot_cells) {
  int tt = get_global_id(0);
  av_vels[tt] /= tot_cells;
}
