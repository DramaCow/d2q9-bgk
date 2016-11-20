/* Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<stddef.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include<mpi.h>

#define NSPEEDS         9
#define MASTER          0
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

// struct to hold the parameter values 
typedef struct
{
  int   gx;            // no. of cells in x direction (for entire grid)
  int   gy;            // no. of cells in y direction (for entire grid)

  int   lx;            // no. of cells in x direction
  int   ly;            // no. of cells in y direction

  int   nx;            // no. of cells in x-direction (plus ghosts)
  int   ny;            // no. of cells in y-direction (plus ghosts)

  int   maxIters;      // no. of iterations 
  int   reynolds_dim;  // dimension for Reynolds number 
  float density;       // density per link 
  float accel;         // density redistribution 
  float omega;         // relaxation parameter 
} t_param;

// struct to hold the 'speed' values 
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

// function prototypes

// load params, allocate memory, load obstacles & initialise fluid particle densities 
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr,
               int *coords, int *dims);

// halo exchange only necessary every other timestep
int halo_exchange_pull(const t_param params, t_speed* restrict cells, int *neighbours, float *buf, float *recvbuf);
int halo_exchange_push(const t_param params, t_speed* restrict cells, int *neighbours, float *sendbuf, float *recvbuf);

int gather_av_velocities(float* restrict av_vels, int tt, float tot_u, int tot_cells);

// seperate functions
void accelerate_flow_1(const t_param params, t_speed* cells, int* obstacles, const int accelerating_row);
void accelerate_flow_2(const t_param params, t_speed* cells, int* obstacles, const int accelerating_row);
void timestep_1(const t_param params, const float tot_cells, t_speed* restrict cells, int *restrict obstacles, float* av_vels, int tt);
void timestep_2(const t_param params, const float tot_cells, t_speed* restrict cells, int *restrict obstacles, float* av_vels, int tt);

int write_values(const t_param params, t_speed* cells, int *obstacles, float *av_vels);

// finalise, including freeing up allocated memory 
int finalise(const t_param* params, t_speed** cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

// Sum all the densities in the grid.
// The total should remain constant from one timestep to the next.
float total_density(const t_param params, t_speed* cells);

// calculate Reynolds number 
float calc_reynolds(const t_param params, float average_velocity);

// utility functions 
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

// main program:
// initialise, timestep loop, finalise
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    // name of the input parameter file 
  char*    obstaclefile = NULL; // name of a the input obstacle file 
  t_param  params;              // struct to hold parameter values 
  t_speed* cells     = NULL;    // grid containing fluid densities 
  int*     obstacles = NULL;    // grid indicating which cells are blocked 
  float* av_vels   = NULL;     // a record of the av. velocity computed for each timestep 
  struct timeval timstr;        // structure to hold elapsed time 
  struct rusage ru;             // structure to hold CPU time--system and user 
  double tic, toc;              // doubleing point numbers to calculate elapsed wallclock time 
  double usrtim;                // doubleing point number to record elapsed user CPU time 
  double systim;                // doubleing point number to record elapsed system CPU time 
  float tot_cells = 0.0f;       // number of non-obstacle cells

  int dims[2] = { 0, 0 };       // dimensions (how to split grid)
  int periods[2] = { 1, 1 };    // wrap around?
  MPI_Comm comm_cart;           
  int coords[2];

  // parse the command line 
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  // Initialize MPI environment.
  MPI_Init(&argc, &argv);

  // Check initialization was successful.
  int flag;
  MPI_Initialized(&flag);
  if(flag != 1) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // get process information
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Dims_create(size, 2, dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);
  MPI_Cart_coords(comm_cart, rank, 2, coords);

  // e, n, w, s, ne, nw, sw, se
  int neighbours[8]; // ranks of particular directions
  {
    MPI_Cart_shift(comm_cart, 0,  1, &neighbours[2], &neighbours[0]);
    MPI_Cart_shift(comm_cart, 1,  1, &neighbours[3], &neighbours[1]);

    int dir[2];

    dir[0] = (coords[0] == dims[0] - 1) ? (0) : (coords[0] + 1); 
    dir[1] = (coords[1] == dims[1] - 1) ? (0) : (coords[1] + 1); 
    MPI_Cart_rank(comm_cart, dir, &neighbours[4]);

    dir[0] = (coords[0] == 0) ? (dims[0] - 1) : (coords[0] - 1);
    dir[1] = (coords[1] == dims[1] - 1) ? (0) : (coords[1] + 1); 
    MPI_Cart_rank(comm_cart, dir, &neighbours[5]);

    dir[0] = (coords[0] == 0) ? (dims[0] - 1) : (coords[0] - 1);
    dir[1] = (coords[1] == 0) ? (dims[1] - 1) : (coords[1] - 1); 
    MPI_Cart_rank(comm_cart, dir, &neighbours[6]);

    dir[0] = (coords[0] == dims[0] - 1) ? (0) : (coords[0] + 1); 
    dir[1] = (coords[1] == 0) ? (dims[1] - 1) : (coords[1] - 1); 
    MPI_Cart_rank(comm_cart, dir, &neighbours[7]);
  }
  printf("rank[%d, %d : %d] --> (%d, %d, %d, %d, %d, %d, %d, %d)\n", coords[0], coords[1], rank, 
  neighbours[0], neighbours[1], neighbours[2], neighbours[3], neighbours[4], neighbours[5], neighbours[6], neighbours[7]);

  // initialise our data structures and load values from file 
  initialise(paramfile, obstaclefile, &params, &cells, &obstacles, &av_vels, coords, dims);

  int rx = params.gx % dims[0];
  int ry = params.gy % dims[1];

  int sx = coords[0] * (params.gx / dims[0])   // starting x position
           + (coords[0] < rx ? coords[0] : rx); 
  int sy = coords[1] * (params.gy / dims[1])   // starting y position
           + (coords[1] < ry ? coords[1] : ry); 

  printf("rank: %d, sx = %d, sy %d\n", rank, sx, sy);

  { // pre-count number of non-obstacles
    float local_tot_cells = 0.0f;
    for (int ii = 0; ii < params.lx * params.ly; ii++) {
      if (!obstacles[ii]) ++local_tot_cells;
    }
    MPI_Reduce(&local_tot_cells, &tot_cells, 1, MPI_FLOAT, MPI_SUM, MASTER, comm_cart);
  }

  // buffer to hold the data dependencies to/from neighbouring segments
	float *sendbuf = (float*)malloc(sizeof(float) * (6 * (params.nx + params.ny)));
  float *recvbuf = (float*)malloc(sizeof(float) * (6 * (params.nx + params.ny)));

  // iterate for maxIters timesteps 
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // NOTE: Whilst this program iterates in strides of 2, it is
  //       trivial to extend the program to be able to end on an odd iteration.
  //       We simply add:
  //        *  a condition check to see if we've ended on an odd iteration
  //        *  compute using the odd timestep functions, taking note that a
  //           cell's speeds are now stored in its neighbours rather than stored locally
  //        *  adjust the calc_reynolds and write values functions to read a cell's speeds
  //           from neighbouring cells rather than from the local cell
  //       I've ommitted these due to time constraints under the assumption that (for now)
  //       inputs will have an even number of max iterations.  
  if (sy <= params.gy - 2 && params.gy - 2 < sy + params.ly) {
    int accelerating_row = (params.gy - 2) - sy;
    printf("(%d, %d) of %d -- [%d, %d]\n", coords[0], coords[1], rank, sy, params.ly);
    for (int tt = 0; tt < params.maxIters; tt+=2) {
      accelerate_flow_1(params, cells, obstacles, accelerating_row);
      halo_exchange_pull(params, cells, neighbours, sendbuf, recvbuf);
      timestep_1(params, tot_cells, cells, obstacles, av_vels, tt);

      accelerate_flow_2(params, cells, obstacles, accelerating_row);
      halo_exchange_push(params, cells, neighbours, sendbuf, recvbuf);
      timestep_2(params, tot_cells, cells, obstacles, av_vels, tt + 1);
    }
  }
  else {
    for (int tt = 0; tt < params.maxIters; tt+=2) {
      halo_exchange_pull(params, cells, neighbours, sendbuf, recvbuf);
      timestep_1(params, tot_cells, cells, obstacles, av_vels, tt);

      halo_exchange_push(params, cells, neighbours, sendbuf, recvbuf);
      timestep_2(params, tot_cells, cells, obstacles, av_vels, tt + 1);
    }
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  free(sendbuf);
  free(recvbuf);

  // === DEFINE T_SPEED MPI TYPE ===
 
  MPI_Datatype mpi_speed_type;
  int blocklen[1] = { NSPEEDS };
  MPI_Aint offsets[1] = { offsetof(t_speed, speeds) };
  MPI_Datatype types[1] = { MPI_FLOAT };
  MPI_Type_create_struct(1, blocklen, offsets, types, &mpi_speed_type);
  MPI_Type_commit(&mpi_speed_type);

  if (rank == MASTER) {
    t_speed *global_cells = (t_speed*)malloc(sizeof(t_speed) * (params.gx * params.gy));
    int *global_obstacles = (int*)malloc(sizeof(int) * (params.gx * params.gy));

    for (int ii = 1; ii < params.ly + 1; ++ii) {
      for (int jj = 1; jj < params.lx + 1; ++jj) {
        for (int kk = 0; kk < NSPEEDS; ++kk) {
          global_cells[(ii - 1) * params.gx + (jj - 1)].speeds[kk] = cells[ii * params.nx + jj].speeds[kk];
        }
        global_obstacles[(ii - 1) * params.gx + (jj - 1)] = obstacles[(ii - 1) * params.lx + (jj - 1)];
      }
    }
    free(cells); 
    free(obstacles);
    cells = global_cells;
    obstacles = global_obstacles;

    // receive segment from each node
    int coords[2], start[2], length[2];
    for (int source = 1; source < size; ++source) {
      MPI_Cart_coords(comm_cart, source, 2, coords);

      int rx = params.gx % dims[0];
      int ry = params.gy % dims[1];

      int sx = coords[0] * (params.gx / dims[0])   // starting x position
               + (coords[0] < rx ? coords[0] : rx); 
      int sy = coords[1] * (params.gy / dims[1])   // starting y position
               + (coords[1] < ry ? coords[1] : ry); 

      int lx = (params.gx / dims[0])
               + (coords[0] < rx ? 1 : 0);
      int ly = (params.gy / dims[1])
               + (coords[1] < ry ? 1 : 0);

      int nx = lx + 2;
      int ny = ly + 2;

      printf("%d : (%d, %d, %d, %d)\n", source, sx, sy, lx, ly);
 
      for (int ii = 0; ii < ly; ++ii) {
				MPI_Recv(&cells[(sy + ii) * params.gx + sx], lx, mpi_speed_type, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&obstacles[(sy + ii) * params.gx + sx], lx, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }
  else {
    for (int ii = 1; ii < params.ly + 1; ++ii) {
      MPI_Ssend(&cells[(ii * params.nx) + 1], params.lx, mpi_speed_type, MASTER, 0, MPI_COMM_WORLD);
      MPI_Ssend(&obstacles[(ii - 1) * params.lx], params.lx, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }
  }

  // write final values and free memory 
  if (rank == MASTER) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, av_vels[params.maxIters - 1]));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels);
  }

  finalise(&params, &cells, &obstacles, &av_vels);

  // Finalize MPI environment
  MPI_Finalize();

  MPI_Finalized(&flag);
  if(flag != 1) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

// =======================
// === OTHER FUNCTIONS ===
// =======================

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr,
               int *coords, int *dims)
{
  char   message[1024];  // message buffer 
  FILE*   fp;            // file pointer 
  int    xx, yy;         // generic array indices 
  int    blocked;        // indicates whether a cell is blocked by an obstacle 
  int    retval;         // to hold return value for checking 

  // open the parameter file 
  fp = fopen(paramfile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  // read in the parameter values 
  retval = fscanf(fp, "%d\n", &(params->gx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->gy));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));
  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));
  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  // and close up the file 
  fclose(fp);

  // === GET PROCESS INFORMATION ===
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rx = params->gx % dims[0];
  int ry = params->gy % dims[1];

  int sx = coords[0] * (params->gx / dims[0])   // starting x position
           + (coords[0] < rx ? coords[0] : rx); 
  int sy = coords[1] * (params->gy / dims[1])   // starting y position
           + (coords[1] < ry ? coords[1] : ry); 

  params->lx = (params->gx / dims[0])
               + (coords[0] < rx ? 1 : 0);
  params->ly = (params->gy / dims[1])
               + (coords[1] < ry ? 1 : 0);

  params->nx = params->lx + 2;
  params->ny = params->ly + 2;

	// === ALLOCATE MEMORY ===

  // main grid 
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->nx * params->ny));
  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  // the map of obstacles 
  *obstacles_ptr = (int*)malloc(sizeof(int) * (params->lx * params->ly));
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

	// === INITIALISE CELL GRID ===

  // initialise densities 
  float w0 = params->density * 4.0f / 9.0f ;
  float w1 = params->density        / 9.0f ;
  float w2 = params->density        / 36.0f;

  for (int ii = 1; ii < params->ly + 1; ii++)
  {
    for (int jj = 1; jj < params->lx + 1; jj++)
    {
      // centre 
      (*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
      // axis directions 
      (*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
      // diagonals 
      (*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
    }
  }

  // === INITIALISE OBSTACLE GRID ===

  // first set all cells in obstacle array to zero 
  for (int ii = 0; ii < params->ly; ii++)
  {
    for (int jj = 0; jj < params->lx; jj++)
    {
      (*obstacles_ptr)[ii * params->lx + jj] = 0;
    }
  }

  // open the obstacle data file 
  fp = fopen(obstaclefile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  // read-in the blocked cells list 
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    // some checks 
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    if (xx < 0 || xx > params->gx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->gy - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    // assign to array (only if within assigned region)
    if ((sx <= xx && xx < sx + params->lx) && 
        (sy <= yy && yy < sy + params->ly)) {
      (*obstacles_ptr)[(yy - sy) * params->lx + (xx - sx)] = blocked;
    }
  }

  // and close the file 
  fclose(fp);

  // allocate space to hold a record of the avarage velocities computed
  // at each timestep
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  // free up allocated memory
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float average_velocity)
{
  const float viscosity = 1.0f / 6.0f * (2.0f / params.omega - 1.0f);
  return average_velocity * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.0f; // accumulator

  ////#pragma omp parallel for default(none) shared(cells) schedule(static) reduction(+:total)
  for (int ii = 0; ii < params.gy; ii++)
  {
    for (int jj = 0; jj < params.gx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii * params.gx + jj].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                       // file pointer 
  const float c_sq = 1.0f / 3.0f; // sq. of speed of sound 
  float local_density;            // per grid cell sum of densities 
  float pressure;                 // fluid pressure in grid cell 
  float u_x;                      // x-component of velocity in grid cell 
  float u_y;                      // y-component of velocity in grid cell 
  float u;                        // norm--root of summed squares--of u_x and u_y 

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.gy; ii++)
  {
    for (int jj = 0; jj < params.gx; jj++)
    {
      // an occupied cell 
      if (obstacles[ii * params.gx + jj])
      {
        u_x = u_y = u = 0.0f;
        pressure = params.density * c_sq;
      }
      // no obstacle 
      else
      {
        local_density = 0.0f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.gx + jj].speeds[kk];
        }

        // compute x velocity component 
        u_x = (cells[ii * params.gx + jj].speeds[1]
               + cells[ii * params.gx + jj].speeds[5]
               + cells[ii * params.gx + jj].speeds[8]
               - (cells[ii * params.gx + jj].speeds[3]
                  + cells[ii * params.gx + jj].speeds[6]
                  + cells[ii * params.gx + jj].speeds[7]))
              / local_density;
        // compute y velocity component 
        u_y = (cells[ii * params.gx + jj].speeds[2]
               + cells[ii * params.gx + jj].speeds[5]
               + cells[ii * params.gx + jj].speeds[6]
               - (cells[ii * params.gx + jj].speeds[4]
                  + cells[ii * params.gx + jj].speeds[7]
                  + cells[ii * params.gx + jj].speeds[8]))
              / local_density;
        // compute norm of velocity 
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        // compute pressure 
        pressure = local_density * c_sq;
      }

      // write to file 
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.gx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

// =====================
// === HALO EXCHANGE ===
// =====================

int halo_exchange_pull(const t_param params, t_speed* restrict cells, int *neighbours, float *sendbuf, float *recvbuf) 
{
  MPI_Request request[8];
  int line;

  // === EDGE ===

  line = params.lx;
  for (int ii = 1; ii < params.ny - 1; ++ii) {
    sendbuf[(ii - 1) * 3    ] = cells[ii * params.nx + line].speeds[5];
    sendbuf[(ii - 1) * 3 + 1] = cells[ii * params.nx + line].speeds[1];
    sendbuf[(ii - 1) * 3 + 2] = cells[ii * params.nx + line].speeds[8];
  }
  line = params.ly;
  for (int jj = 1; jj < params.nx - 1; ++jj) {
    sendbuf[(params.ly * 3) + (jj - 1) * 3    ] = cells[line * params.nx + jj].speeds[6];
    sendbuf[(params.ly * 3) + (jj - 1) * 3 + 1] = cells[line * params.nx + jj].speeds[2];
    sendbuf[(params.ly * 3) + (jj - 1) * 3 + 2] = cells[line * params.nx + jj].speeds[5];
  }
  line = 1;
  for (int ii = 1; ii < params.ny - 1; ++ii) {
    sendbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3    ] = cells[ii * params.nx + line].speeds[6];
    sendbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3 + 1] = cells[ii * params.nx + line].speeds[3];
    sendbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3 + 2] = cells[ii * params.nx + line].speeds[7];
  }
  line = 1;
  for (int jj = 1; jj < params.nx - 1; ++jj) {
    sendbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3    ] = cells[line * params.nx + jj].speeds[7];
    sendbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3 + 1] = cells[line * params.nx + jj].speeds[4];
    sendbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3 + 2] = cells[line * params.nx + jj].speeds[8];
  }

  MPI_Isend(&sendbuf[0], params.ly * 3, MPI_FLOAT, neighbours[0], 0, MPI_COMM_WORLD, &request[0]);
  MPI_Isend(&sendbuf[(params.ly * 3)], params.lx * 3, MPI_FLOAT, neighbours[1], 0, MPI_COMM_WORLD, &request[1]);
  MPI_Isend(&sendbuf[(params.lx * 3 + params.ly * 3)], params.ly * 3, MPI_FLOAT, neighbours[2], 0, MPI_COMM_WORLD, &request[2]);
  MPI_Isend(&sendbuf[(params.lx * 3 + params.ly * 6)], params.lx * 3, MPI_FLOAT, neighbours[3], 0, MPI_COMM_WORLD, &request[3]);

  MPI_Irecv(&recvbuf[0], params.ly * 3, MPI_FLOAT, neighbours[2], 0, MPI_COMM_WORLD, &request[4]);
  MPI_Irecv(&recvbuf[(params.ly * 3)], params.lx * 3, MPI_FLOAT, neighbours[3], 0, MPI_COMM_WORLD, &request[5]);
  MPI_Irecv(&recvbuf[(params.lx * 3 + params.ly * 3)], params.ly * 3, MPI_FLOAT, neighbours[0], 0, MPI_COMM_WORLD, &request[6]);
  MPI_Irecv(&recvbuf[(params.lx * 3 + params.ly * 6)], params.lx * 3, MPI_FLOAT, neighbours[1], 0, MPI_COMM_WORLD, &request[7]);

  MPI_Waitall(8, request, MPI_STATUS_IGNORE);

  line = 0;
  for (int ii = 1; ii < params.ny - 1; ++ii) {
    cells[ii * params.nx + line].speeds[5] = recvbuf[(ii - 1) * 3    ];
    cells[ii * params.nx + line].speeds[1] = recvbuf[(ii - 1) * 3 + 1];
    cells[ii * params.nx + line].speeds[8] = recvbuf[(ii - 1) * 3 + 2];
  }
  line = 0;
  for (int jj = 1; jj < params.nx - 1; ++jj) {
    cells[line * params.nx + jj].speeds[6] = recvbuf[(params.ly * 3) + (jj - 1) * 3    ];
    cells[line * params.nx + jj].speeds[2] = recvbuf[(params.ly * 3) + (jj - 1) * 3 + 1];
    cells[line * params.nx + jj].speeds[5] = recvbuf[(params.ly * 3) + (jj - 1) * 3 + 2];
  }
  line = params.lx + 1;
  for (int ii = 1; ii < params.ny - 1; ++ii) {
    cells[ii * params.nx + line].speeds[6] = recvbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3    ];
    cells[ii * params.nx + line].speeds[3] = recvbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3 + 1];
    cells[ii * params.nx + line].speeds[7] = recvbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3 + 2];
  }
  line = params.ly + 1;
  for (int jj = 1; jj < params.nx - 1; ++jj) {
    cells[line * params.nx + jj].speeds[7] = recvbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3    ];
    cells[line * params.nx + jj].speeds[4] = recvbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3 + 1];
    cells[line * params.nx + jj].speeds[8] = recvbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3 + 2];
  }

  // === CORNERS ===

  MPI_Sendrecv(&cells[params.ly * params.nx + params.lx].speeds[5], 1, MPI_FLOAT, neighbours[4], 0, 
               &cells[0 * params.nx + 0].speeds[5],                 1, MPI_FLOAT, neighbours[6], 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&cells[params.ly * params.nx + 1].speeds[6],       1, MPI_FLOAT, neighbours[5], 0, 
               &cells[0 * params.nx + (params.lx + 1)].speeds[6], 1, MPI_FLOAT, neighbours[7], 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Sendrecv(&cells[1 * params.nx + 1].speeds[7],                             1, MPI_FLOAT, neighbours[6], 0, 
               &cells[(params.ly + 1) * params.nx + (params.lx + 1)].speeds[7], 1, MPI_FLOAT, neighbours[4], 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&cells[1 * params.nx + params.lx].speeds[8],       1, MPI_FLOAT, neighbours[7], 0, 
               &cells[(params.ly + 1) * params.nx + 0].speeds[8], 1, MPI_FLOAT, neighbours[5], 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return EXIT_SUCCESS;
}

int halo_exchange_push(const t_param params, t_speed* restrict cells, int *neighbours, float *sendbuf, float *recvbuf) 
{
  MPI_Request request[8];
  int line;

  // === EDGES === 

  line = params.lx + 1;
  for (int ii = 1; ii < params.ny - 1; ++ii) {
    sendbuf[(ii - 1) * 3    ] = cells[ii * params.nx + line].speeds[6];
    sendbuf[(ii - 1) * 3 + 1] = cells[ii * params.nx + line].speeds[3];
    sendbuf[(ii - 1) * 3 + 2] = cells[ii * params.nx + line].speeds[7];
  }
  line = params.ly + 1;
  for (int jj = 1; jj < params.nx - 1; ++jj) {
    sendbuf[(params.ly * 3) + (jj - 1) * 3    ] = cells[line * params.nx + jj].speeds[7];
    sendbuf[(params.ly * 3) + (jj - 1) * 3 + 1] = cells[line * params.nx + jj].speeds[4];
    sendbuf[(params.ly * 3) + (jj - 1) * 3 + 2] = cells[line * params.nx + jj].speeds[8];
  }
  line = 0;
  for (int ii = 1; ii < params.ny - 1; ++ii) {
    sendbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3    ] = cells[ii * params.nx + line].speeds[5];
    sendbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3 + 1] = cells[ii * params.nx + line].speeds[1];
    sendbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3 + 2] = cells[ii * params.nx + line].speeds[8];
  }
  line = 0;
  for (int jj = 1; jj < params.nx - 1; ++jj) {
    sendbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3    ] = cells[line * params.nx + jj].speeds[6];
    sendbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3 + 1] = cells[line * params.nx + jj].speeds[2];
    sendbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3 + 2] = cells[line * params.nx + jj].speeds[5];
  }

  // corners
  MPI_Isend(&sendbuf[0], params.ly * 3, MPI_FLOAT, neighbours[0], 0, MPI_COMM_WORLD, &request[0]);
  MPI_Isend(&sendbuf[(params.ly * 3)], params.lx * 3, MPI_FLOAT, neighbours[1], 0, MPI_COMM_WORLD, &request[1]);
  MPI_Isend(&sendbuf[(params.lx * 3 + params.ly * 3)], params.ly * 3, MPI_FLOAT, neighbours[2], 0, MPI_COMM_WORLD, &request[2]);
  MPI_Isend(&sendbuf[(params.lx * 3 + params.ly * 6)], params.lx * 3, MPI_FLOAT, neighbours[3], 0, MPI_COMM_WORLD, &request[3]);
  // edges
  /*
  MPI_Isend(&cells[(params.ly + 1) * params.nx + (params.lx + 1)].speeds[7], 1, MPI_FLOAT, neighbours[4], 0, MPI_COMM_WORLD, &request[4]);
  MPI_Isend(&cells[(params.ly + 1) * params.nx + 0].speeds[8], 1, MPI_FLOAT, neighbours[5], 0, MPI_COMM_WORLD, &request[5]);
  MPI_Isend(&cells[0 * params.nx + 0].speeds[5], 1, MPI_FLOAT, neighbours[6], 0, MPI_COMM_WORLD, &request[6]);
  MPI_Isend(&cells[0 * params.nx + (params.lx + 1)].speeds[6], 1, MPI_FLOAT, neighbours[7], 0, MPI_COMM_WORLD, &request[7]);
  */

  // edges
  MPI_Irecv(&recvbuf[0], params.ly * 3, MPI_FLOAT, neighbours[2], 0, MPI_COMM_WORLD, &request[4]);
  MPI_Irecv(&recvbuf[(params.ly * 3)], params.lx * 3, MPI_FLOAT, neighbours[3], 0, MPI_COMM_WORLD, &request[5]);
  MPI_Irecv(&recvbuf[(params.lx * 3 + params.ly * 3)], params.ly * 3, MPI_FLOAT, neighbours[0], 0, MPI_COMM_WORLD, &request[6]);
  MPI_Irecv(&recvbuf[(params.lx * 3 + params.ly * 6)], params.lx * 3, MPI_FLOAT, neighbours[1], 0, MPI_COMM_WORLD, &request[7]);
  // corners
  /*
  MPI_Irecv(&cells[1 * params.nx + 1].speeds[7], 1, MPI_FLOAT, neighbours[6], 0, MPI_COMM_WORLD, &request[12]);
  MPI_Irecv(&cells[1 * params.nx + params.lx].speeds[8], 1, MPI_FLOAT, neighbours[7], 0, MPI_COMM_WORLD, &request[13]);
  MPI_Irecv(&cells[params.ly * params.nx + params.lx].speeds[5], 1, MPI_FLOAT, neighbours[4], 0, MPI_COMM_WORLD, &request[14]);
  MPI_Irecv(&cells[params.ly * params.nx + 1].speeds[6], 1, MPI_FLOAT, neighbours[5], 0, MPI_COMM_WORLD, &request[15]);
  */

  MPI_Waitall(8, request, MPI_STATUS_IGNORE);

  line = 1;
  for (int ii = 1; ii < params.ny - 1; ++ii) {
    cells[ii * params.nx + line].speeds[6] = recvbuf[(ii - 1) * 3    ];
    cells[ii * params.nx + line].speeds[3] = recvbuf[(ii - 1) * 3 + 1];
    cells[ii * params.nx + line].speeds[7] = recvbuf[(ii - 1) * 3 + 2];
  }
  line = 1;
  for (int jj = 1; jj < params.nx - 1; ++jj) {
    cells[line * params.nx + jj].speeds[7] = recvbuf[(params.ly * 3) + (jj - 1) * 3    ];
    cells[line * params.nx + jj].speeds[4] = recvbuf[(params.ly * 3) + (jj - 1) * 3 + 1];
    cells[line * params.nx + jj].speeds[8] = recvbuf[(params.ly * 3) + (jj - 1) * 3 + 2];
  }
  line = params.lx;
  for (int ii = 1; ii < params.ny - 1; ++ii) {
    cells[ii * params.nx + line].speeds[5] = recvbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3    ];
    cells[ii * params.nx + line].speeds[1] = recvbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3 + 1];
    cells[ii * params.nx + line].speeds[8] = recvbuf[(params.lx * 3 + params.ly * 3) + (ii - 1) * 3 + 2];
  }
  line = params.ly;
  for (int jj = 1; jj < params.nx - 1; ++jj) {
    cells[line * params.nx + jj].speeds[6] = recvbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3    ];
    cells[line * params.nx + jj].speeds[2] = recvbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3 + 1];
    cells[line * params.nx + jj].speeds[5] = recvbuf[(params.lx * 3 + params.ly * 6) + (jj - 1) * 3 + 2];
  }

MPI_Sendrecv(&cells[(params.ly + 1) * params.nx + (params.lx + 1)].speeds[7], 1, MPI_FLOAT, neighbours[4], 0, 
               &cells[1 * params.nx + 1].speeds[7],                             1, MPI_FLOAT, neighbours[6], 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&cells[(params.ly + 1) * params.nx + 0].speeds[8], 1, MPI_FLOAT, neighbours[5], 0, 
               &cells[1 * params.nx + params.lx].speeds[8],       1, MPI_FLOAT, neighbours[7], 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&cells[0 * params.nx + 0].speeds[5],                 1, MPI_FLOAT, neighbours[6], 0, 
               &cells[params.ly * params.nx + params.lx].speeds[5], 1, MPI_FLOAT, neighbours[4], 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&cells[0 * params.nx + (params.lx + 1)].speeds[6], 1, MPI_FLOAT, neighbours[7], 0, 
               &cells[params.ly * params.nx + 1].speeds[6],       1, MPI_FLOAT, neighbours[5], 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return EXIT_SUCCESS;
}

int gather_av_velocities(float* restrict av_vels, int tt, float local_tot_u, int tot_cells) {
  float tot_u;
  MPI_Reduce(&local_tot_u, &tot_u, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  av_vels[tt] = tot_u / tot_cells;

  return EXIT_SUCCESS;
}

// ===========================
// === SEPERATED FUNCTIONS ===
// ===========================

void accelerate_flow_1(const t_param params, t_speed* cells, int* obstacles, const int accelerating_row)
{
  // compute weighting factors
  const float w1a = params.density * params.accel / 9.0f;
  const float w2a = params.density * params.accel / 36.0f;
  // rows used for accelerating flow
  const int row2 = accelerating_row;

  //#pragma omp for schedule(static)
  for (int jj = 1; jj < params.lx + 1; ++jj)
  {
    // if the cell is not occupied and
    // we don't send a negative density
    if (!obstacles[row2 * params.lx + (jj - 1)]
        && (cells[(row2 + 1) * params.nx + jj].speeds[3] - w1a) > 0.0f
        && (cells[(row2 + 1) * params.nx + jj].speeds[6] - w2a) > 0.0f
        && (cells[(row2 + 1) * params.nx + jj].speeds[7] - w2a) > 0.0f)
    {
      // increase 'east-side' densities 
      cells[(row2 + 1) * params.nx + jj].speeds[1] += w1a;
      cells[(row2 + 1) * params.nx + jj].speeds[5] += w2a;
      cells[(row2 + 1) * params.nx + jj].speeds[8] += w2a;
      // decrease 'west-side' densities 
      cells[(row2 + 1) * params.nx + jj].speeds[3] -= w1a;
      cells[(row2 + 1) * params.nx + jj].speeds[6] -= w2a;
      cells[(row2 + 1) * params.nx + jj].speeds[7] -= w2a;
    }
  }
}

void accelerate_flow_2(const t_param params, t_speed* cells, int* obstacles, const int accelerating_row)
{
  // compute weighting factors
  const float w1a = params.density * params.accel / 9.0f;
  const float w2a = params.density * params.accel / 36.0f;
  // rows used for accelerating flow
  const int row1 = accelerating_row + 1;
  const int row2 = accelerating_row;
  const int row3 = accelerating_row - 1;

  //#pragma omp for schedule(static)
  for (int jj = 1; jj < params.lx + 1; ++jj)
  {
    int x_e = jj + 1;
    int x_w = jj - 1;

    // if the cell is not occupied and
    // we don't send a negative density
    if (!obstacles[row2 * params.lx + (jj - 1)]
        && (cells[(row2 + 1) * params.nx + x_w].speeds[1] - w1a) > 0.0f
        && (cells[(row1 + 1) * params.nx + x_w].speeds[8] - w2a) > 0.0f
        && (cells[(row3 + 1) * params.nx + x_w].speeds[5] - w2a) > 0.0f)
    {
      // increase 'east-side' densities 
      cells[(row2 + 1) * params.nx + x_e].speeds[3] += w1a;
      cells[(row1 + 1) * params.nx + x_e].speeds[7] += w2a;
      cells[(row3 + 1) * params.nx + x_e].speeds[6] += w2a;
      // decrease 'west-side' densities 
      cells[(row2 + 1) * params.nx + x_w].speeds[1] -= w1a;
      cells[(row1 + 1) * params.nx + x_w].speeds[8] -= w2a;
      cells[(row3 + 1) * params.nx + x_w].speeds[5] -= w2a;
    }
  }
}


void timestep_1(const t_param params, const float tot_cells, t_speed* restrict cells, int *restrict obstacles, float* av_vels, int tt)
{
  // collision constants
  const float w[NSPEEDS] = { 4.0f / 9.0f, 
                             1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 
                             1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };
  const int u[NSPEEDS][2] = { {  0,  0 }, {  1,  0 }, {  0,  1 },
                              { -1,  0 }, {  0, -1 }, {  1,  1 },
                              { -1,  1 }, { -1, -1 }, {  1, -1 } };

  // average velocity locals  
  float tot_u = 0.0f; // accumulated magnitudes of velocity for each cell

  // loop over the cells in the grid
  //#pragma omp for schedule(static)
  //for (int ii = 0; ii < params.ny; ++ii)
  for (int ii = 1; ii < params.ly + 1; ++ii)
  {
    for (int jj = 1; jj < params.lx + 1; ++jj)
    {
      if (!obstacles[(ii - 1) * params.lx + (jj - 1)])
      { 
        // =================
        // === PROPAGATE === aka. streaming
        // =================
        // determine indices of axis-direction neighbours
        // respecting periodic boundary conditions (wrap around)
        int y_n = ii + 1;
        int y_s = ii - 1;
        int x_e = jj + 1;
        int x_w = jj - 1;

        // propagate densities to neighbouring cells, following
        // appropriate directions of travel and writing into
        // scratch space grid
        float tmp_speeds[NSPEEDS];
        tmp_speeds[0] = cells[ii  * params.nx + jj ].speeds[0]; // central cell, no movement
        tmp_speeds[1] = cells[ii  * params.nx + x_w].speeds[1]; // west 
        tmp_speeds[2] = cells[y_s * params.nx + jj ].speeds[2]; // south
        tmp_speeds[3] = cells[ii  * params.nx + x_e].speeds[3]; // east
        tmp_speeds[4] = cells[y_n * params.nx + jj ].speeds[4]; // north
        tmp_speeds[5] = cells[y_s * params.nx + x_w].speeds[5]; // south-west
        tmp_speeds[6] = cells[y_s * params.nx + x_e].speeds[6]; // south-east
        tmp_speeds[7] = cells[y_n * params.nx + x_e].speeds[7]; // north-east
        tmp_speeds[8] = cells[y_n * params.nx + x_w].speeds[8]; // north-west 

        // =================
        // === COLLISION === don't consider occupied cells
        // =================
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
          omega_d_equ[kk] = w[kk] * params.omega * local_density * (1.0f + 3.0f*u_kk + 4.5f*u_kk*u_kk - 1.5f*u_sq);
        }

        // relaxation step
        // store cells speeds in adjacent cells
        for (int kk = 0; kk < NSPEEDS; ++kk) {
          tmp_speeds[kk] *= (1.0f - params.omega);
          tmp_speeds[kk] += omega_d_equ[kk];
        }
        cells[ii  * params.nx + jj ].speeds[0] = tmp_speeds[0]; // central cell, no movement
        cells[ii  * params.nx + x_w].speeds[1] = tmp_speeds[3]; // west
        cells[y_s * params.nx + jj ].speeds[2] = tmp_speeds[4]; // south
        cells[ii  * params.nx + x_e].speeds[3] = tmp_speeds[1]; // east
        cells[y_n * params.nx + jj ].speeds[4] = tmp_speeds[2]; // north
        cells[y_s * params.nx + x_w].speeds[5] = tmp_speeds[7]; // south-west
        cells[y_s * params.nx + x_e].speeds[6] = tmp_speeds[8]; // south-east
        cells[y_n * params.nx + x_e].speeds[7] = tmp_speeds[5]; // north-east
        cells[y_n * params.nx + x_w].speeds[8] = tmp_speeds[6]; // north-west

        // accumulate the norm of x- and y- velocity components
        tot_u += sqrtf(u_x * u_x + u_y * u_y);
      }
    }
  }

  gather_av_velocities(av_vels, tt, tot_u, tot_cells);
}

void timestep_2(const t_param params, const float tot_cells, t_speed* restrict cells, int *restrict obstacles, float* av_vels, int tt)
{
  // collision constants
  const float w[NSPEEDS] = { 4.0f / 9.0f, 
                             1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 
                             1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };
  const int u[NSPEEDS][2] = { {  0,  0 }, {  1,  0 }, {  0,  1 },
                              { -1,  0 }, {  0, -1 }, {  1,  1 },
                              { -1,  1 }, { -1, -1 }, {  1, -1 } };

  // average velocity locals  
  float tot_u = 0.0f; // accumulated magnitudes of velocity for each cell

  // loop over the cells in the grid
  //#pragma omp for schedule(static)
  //for (int ii = 0; ii < params.ny; ++ii)
  for (int ii = 1; ii < params.ly + 1; ++ii)
  {
    for (int jj = 1; jj < params.lx + 1; ++jj)
    {
      if (!obstacles[(ii - 1) * params.lx + (jj - 1)])
      { 
        // ===================
        // === "PROPAGATE" === aka. streaming
        // ===================
        // this iteration only uses local speeds (no need to look at neighbours)
        // [[note: there speeds are "facing" inwards]]
        float tmp_speeds[NSPEEDS];
        tmp_speeds[0] = cells[ii * params.nx + jj].speeds[0];
        tmp_speeds[1] = cells[ii * params.nx + jj].speeds[3];
        tmp_speeds[2] = cells[ii * params.nx + jj].speeds[4];
        tmp_speeds[3] = cells[ii * params.nx + jj].speeds[1];
        tmp_speeds[4] = cells[ii * params.nx + jj].speeds[2];
        tmp_speeds[5] = cells[ii * params.nx + jj].speeds[7];
        tmp_speeds[6] = cells[ii * params.nx + jj].speeds[8];
        tmp_speeds[7] = cells[ii * params.nx + jj].speeds[5];
        tmp_speeds[8] = cells[ii * params.nx + jj].speeds[6];

        // =================
        // === COLLISION === don't consider occupied cells
        // =================
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
          d_equ[kk] = w[kk] * params.omega * local_density * (1.0f + 3.0f*u_kk + 4.5f*u_kk*u_kk - 1.5f*u_sq);
        }

        // relaxation step
        // store cells speeds in current cell only
        for (int kk = 0; kk < NSPEEDS; ++kk)
        {
          cells[ii * params.nx + jj].speeds[kk] = (1.0f - params.omega)*tmp_speeds[kk] + d_equ[kk];
        }

        // accumulate the norm of x- and y- velocity components
        tot_u += sqrtf(u_x * u_x + u_y * u_y);
      }
    }
  }

  gather_av_velocities(av_vels, tt, tot_u, tot_cells);
}
