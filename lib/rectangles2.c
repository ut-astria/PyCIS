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
#include <stdlib.h>
#include <math.h>
#include <tgmath.h>
#include <limits.h>
#include <float.h>
#include<string.h>
#include <time.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_cdf.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_qrng.h>
#include<gsl/gsl_sf_gamma.h>
#include<gsl/gsl_sf_trig.h>
#include<sys/mman.h>

#include "rectangles2.h"
#include "constants.h"
#include "misc.h"
#include "tuples.h"
#include "nfa.h"
#include "gradient.h"

/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangle structure ----------------------------*/
/*----------------------------------------------------------------------------*/



/*----------------------------------------------------------------------------*/
/** Copy one rectangle structure to another.
 */
void rect_copy(struct rect * in, struct rect * out)
{
  /* check parameters */
  if( in == NULL || out == NULL ) error("rect_copy: invalid 'in' or 'out'.");

  /* copy values */
  out->x1 = in->x1;
  out->y1 = in->y1;
  out->x2 = in->x2;
  out->y2 = in->y2;
  out->width = in->width;
  out->x = in->x;
  out->y = in->y;
  out->theta = in->theta;
  out->dx = in->dx;
  out->dy = in->dy;
  out->prec = in->prec;
  out->p = in->p;
}



/*----------------------------------------------------------------------------*/
/** Rectangle points iterator.
 
    INITIAL HEADER
 
    The integer coordinates of pixels inside a rectangle are
    iteratively explored. This structure keep track of the process and
    functions ri_ini(), ri_inc(), ri_end(), and ri_del() are used in
    the process. An example of how to use the iterator is as follows:
    \code

      struct rect * rec = XXX; // some rectangle
      rect_iter * i;
      for( i=ri_ini(rec); !ri_end(i); ri_inc(i) )
        {
          // your code, using 'i->x' and 'i->y' as coordinates
        }
      ri_del(i); // delete iterator

    \endcode
    The pixels are explored 'column' by 'column', where we call
    'column' a set of pixels with the same x value that are inside the
    rectangle. The following is an schematic representation of a
    rectangle, the 'column' being explored is marked by colons, and
    the current pixel being explored is 'x,y'.
    \verbatim

              vx[1],vy[1]
                 *   *
                *       *
               *           *
              *               ye
             *                :  *
        vx[0],vy[0]           :     *
               *              :        *
                  *          x,y          *
                     *        :              *
                        *     :            vx[2],vy[2]
                           *  :                *
        y                     ys              *
        ^                        *           *
        |                           *       *
        |                              *   *
        +---> x                      vx[3],vy[3]

    \endverbatim
    The first 'column' to be explored is the one with the smaller x
    value. Each 'column' is explored starting from the pixel of the
    'column' (inside the rectangle) with the smallest y value.

    The four corners of the rectangle are stored in order that rotates
    around the corners at the arrays 'vx[]' and 'vy[]'. The first
    point is always the one with smaller x value.

    'x' and 'y' are the coordinates of the pixel being explored. 'ys'
    and 'ye' are the start and end values of the current column being
    explored. So, 'ys' < 'ye'.

    UPDATE:
    Towards the 3D iteration algorithm, we significantly restructure
    the increment algorithm ri_inc() to perform incrementation in the 
    rotated space (integer-wise) and then project into the original space
    (where the VonGioi/Liu algorithm would interpolate along the columns/rows)

 */

/*
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
*/

/*----------------------------------------------------------------------------*/
/** Free memory used by a rectangle iterator.
 */
void ri_del(rect_iter * iter)
{
  if( iter == NULL ) error("ri_del: NULL iterator.");
  free( (void *) iter );
}


/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.
    See details in \ref rect_iter
 */
int ri_end(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_end: NULL iterator.");

  /* if the current x value is larger than the largest
     x value in the rectangle (vx[2]), we know the full
     exploration of the rectangle is finished. 
     UPDATE KSD3: Exploration reaches end of projected space */
  return (i->xt > i->xspan) && (i->yt > i->yspan) ;
}

/*----------------------------------------------------------------------------*/
/** Project the coordinates from polar to cartsian space, with origin at v0.
 *  Given basis vectors n,t in the rotated frame forming the rotation matrix Q=[n,t] 
 *  and given an input vector vt incrementing along the polar bases in the polar frame, 
 *  we recover the pixel coordinates in the cartesian frame by : 
 *  (vd) = v0 + Q(vt) 
 *   */
void up_all(rect_iter * i)
{
  //x projection
  i->xd = i->vx[0] + ((double)(i->xt)*(i->dl[0])) + ((double)(i->yt)*(i->dl[1]));
  if (i->x != (int) ceil(i->xd)-1) {
    i->x = (int) ceil(i->xd)-1;
    i->update = 1;}
  //y projection
  i->yd = i->vy[0] + ((double)(i->xt)*(i->dn[0])) + ((double)(i->yt)*(i->dn[1]));
  if (i->y != (int) ceil(i->yd)) {
    i->y = (int) ceil(i->yd);
    i->update = 1;}
}

/*----------------------------------------------------------------------------*/
/** Iterate pixels in the polar reference frame for integer step sizes, 
 *  projecting into cartesian coordinates from the origin v0.*/
void ri_inc(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_inc: NULL iterator.");
  i->update=0;
  //Increment integer-wise in the rotated plane 
  while(i->update==0)
  {
    // if not at end of exploration, inc. y to next 'col'
    if(!(ri_end(i))) {i->yt++;}
    //if at end of col, inc. x to next row
    while( (i->yt > i->yspan) && !(ri_end(i)) )
    {
      i->xt++;    
      if(ri_end(i)) {up_all(i); return;}
      //set yt to zero. 'ys' is f(v[0],xt)
      i->yt = 0; 
    }
    //Update cartesian pixel coordinate 
    up_all(i);
  }
}

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.
    See details in \ref rect_iter
 */
rect_iter * ri_ini(struct rect * r)
{
  double vx[4],vy[4];
  int n,offset;
  rect_iter * i;

  /* check parameters */
  if( r == NULL ) error("ri_ini: invalid rectangle.");

  /* get memory */
  i = (rect_iter *) malloc(sizeof(rect_iter));
  if( i == NULL ) error("ri_ini: Not enough memory.");

  /* build list of rectangle corners ordered
     in a circular way around the rectangle */
  vx[0] = r->x1 - r->dy * r->width / 2.0;
  vy[0] = r->y1 + r->dx * r->width / 2.0;
  vx[1] = r->x2 - r->dy * r->width / 2.0;
  vy[1] = r->y2 + r->dx * r->width / 2.0;
  vx[2] = r->x2 + r->dy * r->width / 2.0;
  vy[2] = r->y2 - r->dx * r->width / 2.0;
  vx[3] = r->x1 + r->dy * r->width / 2.0;
  vy[3] = r->y1 - r->dx * r->width / 2.0;

  /* compute rotation of index of corners needed so that the first
     point has the smaller x.

     if one side is vertical, thus two corners have the same smaller x
     value, the one with the largest y value is selected as the first.
   */
  if( r->x1 < r->x2 && r->y1 <= r->y2 ) offset = 0;
  else if( r->x1 >= r->x2 && r->y1 < r->y2 ) offset = 1;
  else if( r->x1 > r->x2 && r->y1 >= r->y2 ) offset = 2;
  else offset = 3;

  /* apply rotation of index. */
  for(n=0; n<4; n++)
    {
      i->vx[n] = vx[(offset+n)%4];
      i->vy[n] = vy[(offset+n)%4];
    }

  /* Set an initial condition.

     The values are set to values that will cause 'ri_inc' (that will
     be called immediately) to initialize correctly the first 'column'
     and compute the limits 'ys' and 'ye'.

     'y' is set to the integer value of vy[0], the starting corner.

     'ys' and 'ye' are set to very small values, so 'ri_inc' will
     notice that it needs to start a new 'column'.

     The smallest integer coordinate inside of the rectangle is
     'ceil(vx[0])'. The current 'x' value is set to that value minus
     one, so 'ri_inc' (that will increase x by one) will advance to
     the first 'column'.
   */
 
  i->x  = (int) ceil(i->vx[0]) - 1;
  i->y = (int) ceil(i->vy[0]);
  //store explorable range for use in CLSD3 iteration scheme
  i->xspan =  dist(r->x1,r->y1,r->x2,r->y2); 
  i->yspan = r->width;
  i->xt = 0; i->yt =  0;
  i->xd = 0; i->yd = 0.;
  /* Instantiate unit normal and tangent vectors, 
   * fixing the normal vector to the 1st quadrent 
   * to ensure proper exploration along the x (length) vector
   * from the origin at v0, the smallest x coordinate
   */
  i->dl[0] = fabs(r->dx);
  i->dl[1] = fabs(r->dy);
  i->dn[0] = -1.*fabs(r->dy);
  i->dn[1] = fabs(r->dx);
  i->ys=i->ye = -DBL_MAX; //legacy variable 
  /* Correct the orientation of the tangent axis as needed
   * to ensure valid exploration, since v0 should have 
   * the smallest y value, and v2 the largest. 
   * */
  if(i->vy[0] < i->vy[2]){i->dn[0]*=-1.; i->dn[1]*=-1.;} 
  /* advance to the first pixel */
  ri_inc(i);
  return i;
}

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
double rect_nfa(struct rect * rec, image_double angles, double logNT,double *image,int N,int minreg)
{
  rect_iter * i;
  int pts = 0;
  int alg = 0;

  /* check parameters */
  if( rec == NULL ) error("rect_nfa: invalid rectangle.");
  if( angles == NULL ) error("rect_nfa: invalid 'angles'.");

  /* compute the total number of pixels and of aligned points in 'rec' */
  for(i=ri_ini(rec); !ri_end(i); ri_inc(i)) /* rectangle iterator */
    if( i->x >= 0 && i->y >= 0 &&
        i->x < (int) angles->xsize && i->y < (int) angles->ysize )
      {
        ++pts; /* total number of pixels counter */
        if( isaligned(i->x, i->y, angles, rec->theta, rec->prec) )
          ++alg; /* aligned points counter */
      }
  ri_del(i); /* delete iterator */
  if(pts<minreg)
      return -1;
  else
  return nfa(pts,alg,rec->p,logNT,image,N); /* compute NFA value */
}

