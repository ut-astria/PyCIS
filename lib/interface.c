/*----------------------------------------------------------------------------
PyCIS - Python Computational Inference from Structure

    A-contrario inference of object trajectories from structure-in-noise, 
    building on Line Segment Detection (LSD) for dense electro-optical time-series data
    formatted as 3D data cubes, with markov kernel estimation for non-uniform noise models.
    LSD C-extension module equipped with multi-layer a-contrario inference for center-line features
    from gradient information.  Python modules provided for inference of feature classifications
    using second-order gestalts, and ingesting/plotting of FITS-format data files.

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group

**NOTICE: For copyright and licensing, see header for pycis.c 
          and 'notices' at bottom of README

------------------------------------------------------------------------------*/   

/*----------------------------------------------------------------------------*/
/*---------------------------------- Import ---------------------------------*/
/*----------------------------------------------------------------------------*/



#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <tgmath.h>
#include <limits.h>
#include <float.h>
#include<string.h>
#include <time.h>
#include <sys/mman.h>

#include "interface.h"
#include "constants.h"
#include "misc.h"
#include "tuples.h"
#include "markov.h"
#include "lsd2.h"
#include "lsd3.h"
#include "lsd3b.h"

/*----------------------------------------------------------------------------*/
/*----------------------- LSD-WRAPPER INTERFACES -----------------------------*/
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/** LSD Simple Interface.
 */
double * lsd(int * n_out, double * img, int X, int Y,double *inputv, double inputv_size)
{
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);
  
  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */

  return LineSegmentDetection( n_out, img, X, Y,
                              ang_th, log_eps, density_th, n_bins,
                              NULL,NULL,NULL ,inputv,inputv_size);
}

/*----------------------------------------------------------------------------*/
/** LSD3 Simple Interface.
 */
double * lsd3(int * n_out, double * img, int X, int Y, int Z, 
              double *inputv, double inputv_size,double * inputvorth)
{ 
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);

  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */
  return LineSegmentDetection3( n_out, img, X, Y, Z,
                              ang_th, log_eps, density_th, n_bins,
                              NULL,NULL,NULL,NULL ,inputv,inputv_size,inputvorth);
}

/*----------------------------------------------------------------------------*/
/** LSD3 Simple Interface.
 */
double * lsd3b(int * n_out, double * img, int X, int Y, int Z, 
              double *inputv, double inputv_size,double * inputvorth)
{ 
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);

  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */
  return LineSegmentDetection3b( n_out, img, X, Y, Z,
                              ang_th, log_eps, density_th, n_bins,
                              NULL,NULL,NULL,NULL ,inputv,inputv_size,inputvorth);
}

/*----------------------------------------------------------------------------*/
/** LSD3 Centerline Simple Interface.
 */
double * lsd3center(int * n_out, 
               double * img, int X, int Y, int Z,
               double * img0, int X0, int Y0, int Z0,
               double *inputv, double inputv_size,double * inputvorth)
{ 
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);

  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */
  return LineSegmentDetection3Center( n_out, img, X, Y, Z, img0, X0, Y0, Z0,
                              ang_th, log_eps, density_th, n_bins,
                              NULL,NULL,NULL,NULL ,inputv,inputv_size,inputvorth);
}

/*----------------------------------------------------------------------------*/
/** LSD3 Centerline Simple Interface.
 */
double * lsd3centerb(int * n_out, 
               double * img, int X, int Y, int Z,
               double * img0, int X0, int Y0, int Z0,
               double *inputv, double inputv_size,double * inputvorth)
{ 
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);

  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */
  return LineSegmentDetection3Centerb( n_out, img, X, Y, Z, img0, X0, Y0, Z0,
                              ang_th, log_eps, density_th, n_bins,
                              NULL,NULL,NULL,NULL ,inputv,inputv_size,inputvorth);
}

/*----------------------------------------------------------------------------*/
/** LSD+Markov Simple Interface.
 */
double * lsdM(int * n_out, 
               double * img, int X, int Y,
               double * img0, int X0, int Y0,
               double *inputv, double inputv_size)
{
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);
  
  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */

  make_markov(img0, X0, Y0, ang_th, n_bins,inputv,inputv_size);                     
  return LineSegmentDetection( n_out, img, X, Y,
                               ang_th, log_eps, density_th, n_bins,
                              NULL,NULL,NULL ,inputv,inputv_size);
}

/*----------------------------------------------------------------------------*/
/** LSD3+Markov Simple Interface.
 */
double * lsd3M(int * n_out, 
               double * img, int X, int Y, int Z,
               double * img0, int X0, int Y0, int Z0,
               double *inputv, double inputv_size,double * inputvorth)
{

  //Instantiate input variables
  double ang_th;   /* Gradient angle tolerance in degrees.           */
  ang_th=inputv[4];
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  log_eps=-log10(inputv[1]);
  double density_th = 0.0;  /* Minimal density of region points in rectangle. */
  density_th=inputv[2];
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */

  //Instantiate constants
  int i,j,il; //iteration variables
    double start,end; //timing variables
  double sizenum = 5.*sqrt(
    (double)X0*(double)X0 + (double)Y0*(double)Y0 + (double)Z0*(double)Z0);
  if(sizenum>pow(10.,6)) sizenum=pow(10.,6);
  inputv[3] = sizenum; //maximum region size
  //printf("IMSIZE: %d\n",(int)sizenum);fflush(stdout);

  //Caluculate "edge feature" Markov kernel from naive image 
  start=omp_get_wtime();
  make_markov3( img0, X0, Y0, Z0, ang_th, n_bins, inputv,inputv_size,0);
  end=omp_get_wtime();
  printf("MAKEMARKOV: %f seconds\n",end-start);fflush(stdout);
  printf("PARAKernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n\n",
      inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);fflush(stdout);
  //t = clock();
  
  //Calculate "centerline feature" Markov kernel from naive image
  start=omp_get_wtime();
  make_markov3( img0, X0, Y0, Z0, ang_th, n_bins, inputvorth,inputv_size,1);
  end=omp_get_wtime();
  printf("MAKEMARKOVORTH: %f seconds\n",end-start);fflush(stdout);
  printf("ORTHKernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n\n",
      inputvorth[5],inputvorth[6], inputvorth[7],inputvorth[8],inputvorth[9],inputvorth[10]);fflush(stdout);
  
  //Launch LSD3 algorithm with parallel and orthogonal markov kenrnels

  return LineSegmentDetection3( n_out, img, X, Y, Z,
                                ang_th, log_eps, density_th, n_bins,
                                NULL,NULL,NULL,NULL ,inputv,inputv_size,inputvorth);

}