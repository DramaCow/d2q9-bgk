#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  double speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            double density, double accel)
{
  /* compute weighting factors */
  double w1 = density * accel / 9.0;
  double w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[ii * nx + jj].speeds[1] += w1;
    cells[ii * nx + jj].speeds[5] += w2;
    cells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii * nx + jj].speeds[3] -= w1;
    cells[ii * nx + jj].speeds[6] -= w2;
    cells[ii * nx + jj].speeds[7] -= w2;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii  * nx + jj ].speeds[0] = cells[ii * nx + jj].speeds[0]; /* central cell, no movement */
  tmp_cells[ii  * nx + x_e].speeds[1] = cells[ii * nx + jj].speeds[1]; /* east */
  tmp_cells[y_n * nx + jj ].speeds[2] = cells[ii * nx + jj].speeds[2]; /* north */
  tmp_cells[ii  * nx + x_w].speeds[3] = cells[ii * nx + jj].speeds[3]; /* west */
  tmp_cells[y_s * nx + jj ].speeds[4] = cells[ii * nx + jj].speeds[4]; /* south */
  tmp_cells[y_n * nx + x_e].speeds[5] = cells[ii * nx + jj].speeds[5]; /* north-east */
  tmp_cells[y_n * nx + x_w].speeds[6] = cells[ii * nx + jj].speeds[6]; /* north-west */
  tmp_cells[y_s * nx + x_w].speeds[7] = cells[ii * nx + jj].speeds[7]; /* south-west */
  tmp_cells[y_s * nx + x_e].speeds[8] = cells[ii * nx + jj].speeds[8]; /* south-east */
}

kernel void rebound(global t_speed* cells, 
                    global t_speed* tmp_cells, 
                    global int* obstacles,
                    int nx, int ny)
{
  int ii = get_global_id(1);
  int jj = get_global_id(0);

  if (obstacles[ii * nx + jj])
  {
    cells[ii * nx + jj].speeds[1] = tmp_cells[ii * nx + jj].speeds[3];
    cells[ii * nx + jj].speeds[2] = tmp_cells[ii * nx + jj].speeds[4];
    cells[ii * nx + jj].speeds[3] = tmp_cells[ii * nx + jj].speeds[1];
    cells[ii * nx + jj].speeds[4] = tmp_cells[ii * nx + jj].speeds[2];
    cells[ii * nx + jj].speeds[5] = tmp_cells[ii * nx + jj].speeds[7];
    cells[ii * nx + jj].speeds[6] = tmp_cells[ii * nx + jj].speeds[8];
    cells[ii * nx + jj].speeds[7] = tmp_cells[ii * nx + jj].speeds[5];
    cells[ii * nx + jj].speeds[8] = tmp_cells[ii * nx + jj].speeds[6];
  }
}
