
#include "d2q9-bgk.c.opari.inc"
/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
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
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>

#define NSPEEDS         9
#define HALF (NSPEEDS-1)/2
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  double density;       /* density per link */
  double accel;         /* density redistribution */
  double omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  double speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr);

// The main calculation methods.
int accelerate_flow_1(const t_param params, t_speed *restrict cells, int *restrict obstacles);
int accelerate_flow_2(const t_param params, t_speed *restrict cells, int *restrict obstacles);
double propagate_collide_1(const t_param params, t_speed *restrict cells, int *restrict obstacles);
double propagate_collide_2(const t_param params, t_speed *restrict cells, int *restrict obstacles);

int write_values(const t_param params, t_speed *cells, int *obstacles, double *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity_1(const t_param params, t_speed *restrict cells, int *restrict obstacles);
double av_velocity_2(const t_param params, t_speed *restrict cells, int *restrict obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  double* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (int tt = 0; tt < params.maxIters; tt+=2)
  {
  	accelerate_flow_1(params, cells, obstacles);
  	propagate_collide_1(params, cells, obstacles);
    av_vels[tt] = av_velocity_1(params, cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif

  	accelerate_flow_2(params, cells, obstacles);
  	propagate_collide_2(params, cells, obstacles);
    av_vels[tt+1] = av_velocity_2(params, cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt+1);
    printf("av velocity: %.12E\n", av_vels[tt+1]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

// ----------------------------------------------------------------
//
int accelerate_flow_1(const t_param params, t_speed *restrict cells, int *restrict obstacles)
{
  /* compute weighting factors */
  double w1 = params.density * params.accel / 9.0;
  double w2 = params.density * params.accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = params.ny - 2;

  for (int jj = 0; jj < params.nx; jj++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii * params.nx + jj]
        && (cells[ii * params.nx + jj].speeds[3] - w1) > 0.0
        && (cells[ii * params.nx + jj].speeds[6] - w2) > 0.0
        && (cells[ii * params.nx + jj].speeds[7] - w2) > 0.0)
    {
      /* increase 'east-side' densities */
      cells[ii * params.nx + jj].speeds[1] += w1;
      cells[ii * params.nx + jj].speeds[5] += w2;
      cells[ii * params.nx + jj].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii * params.nx + jj].speeds[3] -= w1;
      cells[ii * params.nx + jj].speeds[6] -= w2;
      cells[ii * params.nx + jj].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

double propagate_collide_1(const t_param params, t_speed *restrict cells, int *restrict obstacles)
{
  // collision locals
  const double w0 = (4.0 / 9.0 )*params.omega; // weighting factor
  const double w1 = (1.0 / 9.0 )*params.omega; // weighting factor
  const double w2 = (1.0 / 36.0)*params.omega; // weighting factor

  // loop over the cells in the grid
  // NB the collision step is called after
  // the propagate step and so values of interest
  // are in the scratch-space grid
  //#pragma omp parallel for default(none) shared(cells,obstacles) private(tmp) schedule(static)
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      if (!obstacles[ii * params.nx + jj])
      { 
	      // =================
	      // === PROPAGATE === aka. streaming
	      // =================
	      // determine indices of axis-direction neighbours
	      // respecting periodic boundary conditions (wrap around)
	      int y_n = (ii + 1) % params.ny;
	      int x_e = (jj + 1) % params.nx;
	      int y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);
	      int x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);

	      // propagate densities to neighbouring cells, following
	      // appropriate directions of travel and writing into
	      // scratch space grid
  			t_speed tmp;
	      tmp.speeds[0] = cells[ii  * params.nx + jj ].speeds[0]; // central cell, no movement
	      tmp.speeds[1] = cells[ii  * params.nx + x_w].speeds[1]; // west 
	      tmp.speeds[2] = cells[y_s * params.nx + jj ].speeds[2]; // south
	      tmp.speeds[3] = cells[ii  * params.nx + x_e].speeds[3]; // east
	      tmp.speeds[4] = cells[y_n * params.nx + jj ].speeds[4]; // north
	      tmp.speeds[5] = cells[y_s * params.nx + x_w].speeds[5]; // south-west
	      tmp.speeds[6] = cells[y_s * params.nx + x_e].speeds[6]; // south-east
	      tmp.speeds[7] = cells[y_n * params.nx + x_e].speeds[7]; // north-east
	      tmp.speeds[8] = cells[y_n * params.nx + x_w].speeds[8]; // north-west 

      	// =================
      	// === COLLISION === don't consider occupied cells
      	// =================
        // compute local density total
        double local_density = 0.0;
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp.speeds[kk];
        }

        // compute x velocity component
        double u_x = ( 
											 + tmp.speeds[1]
                       + tmp.speeds[5]
                       + tmp.speeds[8]
                       - tmp.speeds[3]
                       - tmp.speeds[6]
                       - tmp.speeds[7] 
										 ) / local_density;
                     
        // compute y velocity component
        double u_y = ( 
											 + tmp.speeds[2]
                       + tmp.speeds[5]
                       + tmp.speeds[6]
											 - tmp.speeds[4]
                       - tmp.speeds[7]
                       - tmp.speeds[8] 
                     ) / local_density;

        // velocity squared
        double u_sq = 1.5*(u_x * u_x + u_y * u_y);

        // directional velocity components
        double u1 =   u_x;        // east
        double u2 =         u_y;  // north
        double u3 = - u_x;        // west
        double u4 =       - u_y;  // south
        double u5 =   u_x + u_y;  // north-east
        double u6 = - u_x + u_y;  // north-west
        double u7 = - u_x - u_y;  // south-west
        double u8 =   u_x - u_y;  // south-east

        // omega * equilibrium densities
        double omega_d_equ[NSPEEDS];
				// zero velocity density: weight w0
        omega_d_equ[0] = w0 * local_density * (1.0 - u_sq);
        // axis speeds: weight w1
        omega_d_equ[1] = w1 * local_density * (1.0 + u1*(3.0 + 4.5*u1) - u_sq);
        omega_d_equ[2] = w1 * local_density * (1.0 + u2*(3.0 + 4.5*u2) - u_sq);
        omega_d_equ[3] = w1 * local_density * (1.0 + u3*(3.0 + 4.5*u3) - u_sq);
        omega_d_equ[4] = w1 * local_density * (1.0 + u4*(3.0 + 4.5*u4) - u_sq);
        // diagonal speeds: weight w2
        omega_d_equ[5] = w2 * local_density * (1.0 + u5*(3.0 + 4.5*u5) - u_sq);
        omega_d_equ[6] = w2 * local_density * (1.0 + u6*(3.0 + 4.5*u6) - u_sq);
        omega_d_equ[7] = w2 * local_density * (1.0 + u7*(3.0 + 4.5*u7) - u_sq);
        omega_d_equ[8] = w2 * local_density * (1.0 + u8*(3.0 + 4.5*u8) - u_sq);

				// relaxation step
				// store cells speeds in adjacent cells
	      cells[ii  * params.nx + jj ].speeds[0] = (1.0 - params.omega)*tmp.speeds[0] + omega_d_equ[0]; // central cell, no movement
	      cells[ii  * params.nx + x_w].speeds[1] = (1.0 - params.omega)*tmp.speeds[3] + omega_d_equ[3]; // west
	      cells[y_s * params.nx + jj ].speeds[2] = (1.0 - params.omega)*tmp.speeds[4] + omega_d_equ[4]; // south
	      cells[ii  * params.nx + x_e].speeds[3] = (1.0 - params.omega)*tmp.speeds[1] + omega_d_equ[1]; // east
	      cells[y_n * params.nx + jj ].speeds[4] = (1.0 - params.omega)*tmp.speeds[2] + omega_d_equ[2]; // north
	      cells[y_s * params.nx + x_w].speeds[5] = (1.0 - params.omega)*tmp.speeds[7] + omega_d_equ[7]; // south-west
	      cells[y_s * params.nx + x_e].speeds[6] = (1.0 - params.omega)*tmp.speeds[8] + omega_d_equ[8]; // south-east 
	      cells[y_n * params.nx + x_e].speeds[7] = (1.0 - params.omega)*tmp.speeds[5] + omega_d_equ[5]; // north-east
	      cells[y_n * params.nx + x_w].speeds[8] = (1.0 - params.omega)*tmp.speeds[6] + omega_d_equ[6]; // north-west
			}
    }
  }

  return EXIT_SUCCESS;
}

double av_velocity_1(const t_param params, t_speed *restrict cells, int *restrict obstacles)
{
  int    tot_cells = 0;  // no. of cells used in calculation
  double tot_u = 0.0;    // accumulated magnitudes of velocity for each cell

  /* loop over all non-blocked cells */
  //#pragma omp parallel for default(none) shared(cells,obstacles) schedule(static) reduction(+:tot_cells,tot_u)
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
	      // determine indices of axis-direction neighbours
	      // respecting periodic boundary conditions (wrap around)
	      int y_n = (ii + 1) % params.ny;
	      int x_e = (jj + 1) % params.nx;
	      int y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);
	      int x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);

        /* local density total */
        double local_density = (
	      												 + cells[ii  * params.nx + jj ].speeds[0] // central cell, no movement
	     													 + cells[ii  * params.nx + x_w].speeds[1] // west 
	     													 + cells[y_s * params.nx + jj ].speeds[2] // south
	     													 + cells[ii  * params.nx + x_e].speeds[3] // east
	     													 + cells[y_n * params.nx + jj ].speeds[4] // north
	     													 + cells[y_s * params.nx + x_w].speeds[5] // south-west
	     													 + cells[y_s * params.nx + x_e].speeds[6] // south-east
	     													 + cells[y_n * params.nx + x_e].speeds[7] // north-east
	     													 + cells[y_n * params.nx + x_w].speeds[8] // north-west 
															 );

        /* x-component of velocity */
        double u_x = (
											 + cells[ii  * params.nx + x_e].speeds[3]
                       + cells[y_n * params.nx + x_e].speeds[7]
                       + cells[y_s * params.nx + x_e].speeds[6]
											 - cells[ii  * params.nx + x_w].speeds[1]
                       - cells[y_n * params.nx + x_w].speeds[8]
                       - cells[y_s * params.nx + x_w].speeds[5]
										 ) / local_density;
        /* compute y velocity component */
        double u_y = (
											 + cells[y_n * params.nx + jj ].speeds[4]
                       + cells[y_n * params.nx + x_w].speeds[8]
                       + cells[y_n * params.nx + x_e].speeds[7]
											 - cells[y_s * params.nx + jj ].speeds[2]
                       - cells[y_s * params.nx + x_w].speeds[5]
                       - cells[y_s * params.nx + x_e].speeds[6]
										 ) / local_density;
        /* accumulate the norm of x- and y- velocity components */
				tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        tot_cells++;
      }
    }
  }

  return tot_u / (double)tot_cells;
}

// ----------------------------------------------------------------

int accelerate_flow_2(const t_param params, t_speed *restrict cells, int *restrict obstacles)
{
  // compute weighting factors
  double w1 = params.density * params.accel / 9.0;
  double w2 = params.density * params.accel / 36.0;

  // modify the 2nd row of the grid
  int ii = params.ny - 2;
  int y_n = (ii + 1) % params.ny;
  int y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);

  for (int jj = 0; jj < params.nx; jj++)
  {
  	int x_e = (jj + 1) % params.nx;
  	int x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);

    // if the cell is not occupied and
    // we don't send a negative density
    if (!obstacles[ii * params.nx + jj]
        && (cells[ii  * params.nx + x_w].speeds[1] - w1) > 0.0
        && (cells[y_n * params.nx + x_w].speeds[8] - w2) > 0.0
        && (cells[y_s * params.nx + x_w].speeds[5] - w2) > 0.0)
    {
      /* increase 'east-side' densities */
      cells[ii  * params.nx + x_e].speeds[3] += w1;
      cells[y_n * params.nx + x_e].speeds[7] += w2;
      cells[y_s * params.nx + x_e].speeds[6] += w2;
      /* decrease 'west-side' densities */
      cells[ii  * params.nx + x_w].speeds[1] -= w1;
      cells[y_n * params.nx + x_w].speeds[8] -= w2;
      cells[y_s * params.nx + x_w].speeds[5] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

double propagate_collide_2(const t_param params, t_speed *restrict cells, int *restrict obstacles)
{
  // collision locals
  const double w0 = (4.0 / 9.0 )*params.omega; // weighting factor
  const double w1 = (1.0 / 9.0 )*params.omega; // weighting factor
  const double w2 = (1.0 / 36.0)*params.omega; // weighting factor

  // loop over the cells in the grid
  // NB the collision step is called after
  // the propagate step and so values of interest
  // are in the scratch-space grid
  //#pragma omp parallel for default(none) shared(cells,obstacles) private(tmp) schedule(static)
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      if (!obstacles[ii * params.nx + jj])
      { 
	      // ===================
	      // === "PROPAGATE" === aka. streaming
	      // ===================
	      // this iteration only uses local speeds (no need to look at neighbours)
	      // [[note: there speeds are "facing" inwards]]
  			t_speed tmp;
	      tmp.speeds[0] = cells[ii * params.nx + jj].speeds[0];
	      tmp.speeds[1] = cells[ii * params.nx + jj].speeds[3];
	      tmp.speeds[2] = cells[ii * params.nx + jj].speeds[4];
	      tmp.speeds[3] = cells[ii * params.nx + jj].speeds[1];
	      tmp.speeds[4] = cells[ii * params.nx + jj].speeds[2];
	      tmp.speeds[5] = cells[ii * params.nx + jj].speeds[7];
	      tmp.speeds[6] = cells[ii * params.nx + jj].speeds[8];
	      tmp.speeds[7] = cells[ii * params.nx + jj].speeds[5];
	      tmp.speeds[8] = cells[ii * params.nx + jj].speeds[6];

      	// =================
      	// === COLLISION === don't consider occupied cells
      	// =================
        // compute local density total
        double local_density = 0.0;
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp.speeds[kk];
        }

        // compute x velocity component
        double u_x = ( 
											 + tmp.speeds[1]
                       + tmp.speeds[5]
                       + tmp.speeds[8]
                       - tmp.speeds[3]
                       - tmp.speeds[6]
                       - tmp.speeds[7] 
										 ) / local_density;
                     
        // compute y velocity component
        double u_y = ( 
											 + tmp.speeds[2]
                       + tmp.speeds[5]
                       + tmp.speeds[6]
											 - tmp.speeds[4]
                       - tmp.speeds[7]
                       - tmp.speeds[8] 
                     ) / local_density;

        // velocity squared
        double u_sq = 1.5*(u_x * u_x + u_y * u_y);

        // directional velocity components
        double u1 =   u_x;        // east
        double u2 =         u_y;  // north
        double u3 = - u_x;        // west
        double u4 =       - u_y;  // south
        double u5 =   u_x + u_y;  // north-east
        double u6 = - u_x + u_y;  // north-west
        double u7 = - u_x - u_y;  // south-west
        double u8 =   u_x - u_y;  // south-east

        // omega * equilibrium densities
        double omega_d_equ[NSPEEDS];
				// zero velocity density: weight w0
        omega_d_equ[0] = w0 * local_density * (1.0 - u_sq);
        // axis speeds: weight w1
        omega_d_equ[1] = w1 * local_density * (1.0 + u1*(3.0 + 4.5*u1) - u_sq);
        omega_d_equ[2] = w1 * local_density * (1.0 + u2*(3.0 + 4.5*u2) - u_sq);
        omega_d_equ[3] = w1 * local_density * (1.0 + u3*(3.0 + 4.5*u3) - u_sq);
        omega_d_equ[4] = w1 * local_density * (1.0 + u4*(3.0 + 4.5*u4) - u_sq);
        // diagonal speeds: weight w2
        omega_d_equ[5] = w2 * local_density * (1.0 + u5*(3.0 + 4.5*u5) - u_sq);
        omega_d_equ[6] = w2 * local_density * (1.0 + u6*(3.0 + 4.5*u6) - u_sq);
        omega_d_equ[7] = w2 * local_density * (1.0 + u7*(3.0 + 4.5*u7) - u_sq);
        omega_d_equ[8] = w2 * local_density * (1.0 + u8*(3.0 + 4.5*u8) - u_sq);

				// relaxation step
				// store cells speeds in current cell only
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
				  cells[ii * params.nx + jj].speeds[kk] = (1.0 - params.omega)*tmp.speeds[kk] + omega_d_equ[kk];
				}
			}
    }
  }

  return EXIT_SUCCESS;
}

double av_velocity_2(const t_param params, t_speed *restrict cells, int *restrict obstacles)
{
  int    tot_cells = 0;  // no. of cells used in calculation
  double tot_u = 0.0;    // accumulated magnitudes of velocity for each cell

  /* loop over all non-blocked cells */
  //#pragma omp parallel for default(none) shared(cells,obstacles) schedule(static) reduction(+:tot_cells,tot_u)
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        /* local density total */
        double local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* x-component of velocity */
        double u_x = (
											 + cells[ii * params.nx + jj].speeds[1]
                       + cells[ii * params.nx + jj].speeds[5]
                       + cells[ii * params.nx + jj].speeds[8]
											 - cells[ii * params.nx + jj].speeds[3]
                       - cells[ii * params.nx + jj].speeds[6]
                       - cells[ii * params.nx + jj].speeds[7]
										 ) / local_density;
        /* compute y velocity component */
        double u_y = (
											 + cells[ii * params.nx + jj].speeds[2]
                       + cells[ii * params.nx + jj].speeds[5]
                       + cells[ii * params.nx + jj].speeds[6]
											 - cells[ii * params.nx + jj].speeds[4]
                       - cells[ii * params.nx + jj].speeds[7]
                       - cells[ii * params.nx + jj].speeds[8]
										 ) / local_density;
        /* accumulate the norm of x- and y- velocity components */
				tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        tot_cells++;
      }
    }
  }

  return tot_u / (double)tot_cells;
}

// ----------------------------------------------------------------

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  double w0 = params->density * 4.0 / 9.0;
  double w1 = params->density      / 9.0;
  double w2 = params->density      / 36.0;

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

  return av_velocity_2(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
  double total = 0.0; // accumulator

  //#pragma omp parallel for default(none) shared(cells) schedule(static) reduction(+:total)
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii * params.nx + jj].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels)
{
  FILE* fp;                     /* file pointer */
  const double c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  double local_density;         /* per grid cell sum of densities */
  double pressure;              /* fluid pressure in grid cell */
  double u_x;                   /* x-component of velocity in grid cell */
  double u_y;                   /* y-component of velocity in grid cell */
  double u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii * params.nx + jj].speeds[1]
               + cells[ii * params.nx + jj].speeds[5]
               + cells[ii * params.nx + jj].speeds[8]
               - (cells[ii * params.nx + jj].speeds[3]
                  + cells[ii * params.nx + jj].speeds[6]
                  + cells[ii * params.nx + jj].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii * params.nx + jj].speeds[2]
               + cells[ii * params.nx + jj].speeds[5]
               + cells[ii * params.nx + jj].speeds[6]
               - (cells[ii * params.nx + jj].speeds[4]
                  + cells[ii * params.nx + jj].speeds[7]
                  + cells[ii * params.nx + jj].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
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
