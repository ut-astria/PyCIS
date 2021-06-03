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
/*---------------------------- rectangles2.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef RECTANGLES2_HEADER
#define RECTANGLES2_HEADER

#include "tuples.h"

//Define functions
/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct rect
{
  double x1,y1,x2,y2;  /* first and second point of the line segment */
  double width;        /* rectangle width */
  double x,y;          /* center of the rectangle */
  double theta;        /* angle */
  double dx,dy;        /* (dx,dy) is vector oriented as the line segment */
  double prec;         /* tolerance angle */
  double p;            /* probability of a point with angle within 'prec' */
};

void rect_copy(struct rect * in, struct rect * out);

//see rectangles2.c
typedef struct 
{
double vx[4];     // rectangle's corner X coordinates in circular order 
double vy[4];     // rectangle's corner Y coordinates in circular order 
int x,y;          // coordinates of currently explored pixel 
double ys,ye;     //LSDSAR: endpoins of y column at x step
//INTRODUCED FOR LSD3, for projected iteration in polar bases 
int update;       // indicator for if projected pixel is a new coordinate
double xd,yd;     // projected coordinate in cartesian basis
int xt,yt;        // explored coordinate in sphereical basis
int xspan, yspan; // range of explorable space in spherical basis
double dl[2];     //vector for rotating the x coordinate (normal vector)
double dn[2];     //vector for rotating the y coordinate (tangent vector)
} rect_iter;

void ri_del(rect_iter * iter);
int ri_end(rect_iter * i);
void up_all(rect_iter * i);
void ri_inc(rect_iter * i);
rect_iter * ri_ini(struct rect * r);
double rect_nfa(struct rect * rec, image_double angles, 
                        double logNT,double *image,int N,int minreg);
double rect_nfaORTH(struct rect * rec, image_double angles, 
                        double logNT,double *image,double *pset,int N,int minreg);


//Close header
#endif
