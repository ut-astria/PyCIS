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
/*---------------------------- rectangles3.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef RECTANGLES3_HEADER
#define RECTANGLES3_HEADER

#include "misc.h" //required for anlges
#include "gradient.h"

//Define functions
/*----------------------------------------------------------------------------*/
/** Rectangular prism structure: line segment with two orthogonal widths.
 */
struct rect3
{
  double x1,y1,z1,x2,y2,z2;     /* first and second point of the line segment */
  double length,width1,width2;  /* rectangle width */
  double x,y,z;                 /* center of the rectangle */
  angles3 theta;                /* az/el angle as struct angle3 */
  double dl[3],dw1[3],dw2[3];   /* spherical basis vectors of the principal axis*/
  double prec;                  /* tolerance angle */
  double p;                     /* probability of a point with angle within 'prec' */
};

void rect3_copy(struct rect3 * in, struct rect3 * out);

/*See rectangles3.c*/
typedef struct 
{
  double vx[8];  /* rectangle's corner X coordinates in circular order */
  double vy[8];  /* rectangle's corner Y coordinates in circular order */
  double vz[8]; 
  int x,y,z; // pixel coordinates in original image frame
  //INTRODUCED FOR LSD3, for projected iteration in polar bases 
  int update;              // indicator for if projected pixel is a new coordinate
  double xd,yd,zd;         // projected coordinate in cartesian basis
  int xt,yt,zt;            // explored coordinate in sphereical basis
  int xspan, yspan, zspan; // range of explorable space in spherical basis
  double dl[3];            //vector for rotating the x coordinate (normal vector)
  double dw1[3];           //vector for rotating the y coordinate (azimuth tangent)
  double dw2[3];           //vector for rotating the z coordinate (elevation tangent)
} rect3_iter;

void ri3_del(rect3_iter * iter);
int ri3_end(rect3_iter * i);
void up_all3(rect3_iter * i);
void ri3_inc(rect3_iter * i);
rect3_iter * ri3_ini(struct rect3 * r);
double rect3_nfa(struct rect3 * rec, grads angles, 
                        double logNT,double *image,int N,int minreg);
double rect3_nfaORTH(struct rect3 * rec, grads angles, 
                            double logNT,double *image,double *pset, int N,int minreg);


//Close header
#endif
