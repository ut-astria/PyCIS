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
#include<gsl/gsl_sf_gamma.h>
#include<gsl/gsl_sf_trig.h>
#include<sys/mman.h>

#include "lsd2.h"
#include "constants.h"
#include "misc.h"
#include "tuples.h"
#include "gradient.h"
#include "gaussians.h"
#include "markov.h"
#include "regions2.h"
#include "rectangles2.h"

/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Detector ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** 2D LSD Algorithm- restricted to edge detection
 * TODO: Update to centerline output per 3D algorithm.
 *        note updates needed to non-null region handling 
 */
double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y ,double * inputv, double inputv_size)
{
  //initalization 
  image_double image;
  ntuple_list out = new_ntuple_list(7);
  double * return_value;
  image_double angles,modgrad;
  image_char used;
  image_int region = NULL;
  struct coorlist * list_p;
  void * mem_p;
  struct rect rec;
  struct point * reg;
  int reg_size,min_reg_size,i;
  unsigned int xsize,ysize;
  double reg_angle,prec,p,log_nfa,logNT;
  double scale,sigma_scale;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */
  double beta=inputv[0];
  int sizenum=(int) inputv[3];

  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 ) error("invalid image input."); 
  if( ang_th <= 0.0 || ang_th >= 180.0 )
    error("'ang_th' value must be in the range (0,180).");
  if( density_th < 0.0 || density_th > 1.0 )
    error("'density_th' value must be in the range [0,1].");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");

  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0; 


  /* load and scale image (if necessary) and compute angle at each pixel 
   * Added back in VonGioi's gaussian downscaling for non-SAR data  */
  //set scale defaults
  if (inputv_size>11)  scale=(double)inputv[11];
  else scale=1.0;
  if (inputv_size>12)  sigma_scale=(double)inputv[12];
  else sigma_scale=0.6;
  //optional gaussian scaling 
  image = new_image_double_ptr( (unsigned int) X, (unsigned int) Y, img );
  if (scale != 1.0) 
  {
    sigma_scale = (double) inputv[12];
    image_double scaled_image;
    scaled_image = gaussian_sampler(image,scale,sigma_scale);
    angles = ll_angle( scaled_image, &list_p, &mem_p, &modgrad,
		       (unsigned int) n_bins,beta);
    free_image_double(scaled_image);
  }
  else angles = ll_angle( image, &list_p, &mem_p, &modgrad,
		       (unsigned int) n_bins,beta);
//set size variables
  xsize = angles->xsize;
  ysize = angles->ysize;

  /* Number of Tests - NT

     The theoretical number of tests is Np.(XY)^(5/2)
     where X and Y are number of columns and rows of the image.
     Np corresponds to the number of angle precisions considered.
     
     We consider Np=3 per Liu et al, thus,
     the number of tests is 3 * (X*Y)^(5/2)
     whose logarithm value is  log10(3) + 5/2 * (log10(X) + log10(Y)).
  */
  logNT = 5.0 * ( log10( (double) xsize ) + log10( (double) ysize ) ) / 2.0
          + log10(3.0);    

  /* initialize some structures */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL ) /* save region data */
    region = new_image_int_ini(angles->xsize,angles->ysize,0);
  used = new_image_char_ini(xsize,ysize,NOTUSED);
  reg = (struct point *) calloc( (size_t) (xsize*ysize), sizeof(struct point) );
  if( reg == NULL ) error("not enough memory!");
  
  //instantaite markov variables
  double p0, p1, p11, p10, p01, p00;
  int N=sizenum;
  int NOUT=N;
  int min_reg_size_2, min_reg_size_4;
  double * output = ( double *) malloc(N*N*sizeof(double));
  double * output_2 = ( double *) malloc(N*N*sizeof(double));
  double * output_4 = ( double *) malloc(N*N*sizeof(double));

  //solve markov transition matrices using kernels
  p11=inputv[5]; p01=inputv[6];
  p1=p;  p0=1-p1;         
  min_reg_size=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
  NFA_matrix(output,p0,p11,p01,N);
  p11=inputv[7];  p01=inputv[8];
  p1=p/2;  p0=1-p1;
  min_reg_size_2=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
  NFA_matrix(output_2,p0,p11,p01,N);
  p11=inputv[9];  p01=inputv[10];
  p1=p/4;  p0=1-p1;
  min_reg_size_4=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
  NFA_matrix(output_4,p0,p11,p01,N);

  

   /*
   -------------------------------------------------------------------------------------------------------------
   Begin line segment search 
   -------------------------------------------------------------------------------------------------------------
   */



  /* search for line segments */
  for(; list_p != NULL; list_p = list_p->next )
    if( used->data[ list_p->x + list_p->y * used->xsize ] == NOTUSED &&
        angles->data[ list_p->x + list_p->y * angles->xsize ] != NOTDEF )
       /* there is no risk of double comparison problems here
          because we are only interested in the exact NOTDEF value */
      {
        /* find the region of connected point and ~equal angle */
        region_grow( list_p->x, list_p->y, angles, reg, &reg_size,
                     &reg_angle, used, prec );

        /* reject small regions */
        if( reg_size < min_reg_size ) continue;

        /* construct rectangular approximation for the region */
        region2rect(reg,reg_size,modgrad,reg_angle,prec,p,&rec);

        /* Check if the rectangle exceeds the minimal density of
           region points. If not, try to improve the region.
           The rectangle will be rejected if the final one does
           not fulfill the minimal density condition.
           This is an addition to the original LSD algorithm published in
           "LSD: A Fast Line Segment Detector with a False Detection Control"
           by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
           The original algorithm is obtained with density_th = 0.0.
         */
        if( !refine( reg, &reg_size, modgrad, reg_angle,
                     prec, p, &rec, used, angles, density_th ) ) continue;

        /* compute NFA value */
        log_nfa = rect_improve(&rec,angles,logNT,log_eps,output,output_2,output_4,NOUT,min_reg_size,min_reg_size_2,min_reg_size_4);
         
        if( log_nfa <= log_eps ) continue;

        /* A New Line Segment was found! */
        ++ls_count;  /* increase line segment counter */

        /*
           The gradient was computed with a 2x2 mask, its value corresponds to
           points with an offset of (0.5,0.5), that should be added to output.
           The coordinates origin is at the center of pixel (0,0).
           NOTE: Disabled by Liu et al per GR gradient algorithm 
         */
        rec.x1 += 0.; rec.y1 += 0.;
        rec.x2 += 0.; rec.y2 += 0.;

        /*Correct for gaussian scaling */
        if(scale != 1.0)
        {
          rec.x1/=scale; rec.y1/=scale;
          rec.x2/=scale; rec.y2/=scale;
          rec.width/=scale;
        }
        /* add line segment found to output */
        add_7tuple( out, rec.x1, rec.y1, rec.x2, rec.y2,
                         rec.width, rec.p, log_nfa );

        /* add region number to 'region' image if needed */
        if( region != NULL )
          for(i=0; i<reg_size; i++)
            region->data[ reg[i].x + reg[i].y * region->xsize ] = ls_count;
      }
      
  

   /*
   -------------------------------------------------------------------------------------------------------------
   Free memory and return result
   -------------------------------------------------------------------------------------------------------------
   */



  /* only the double_image structure should be freed,
    the data pointer was provided to this functions
    and should not be destroyed.                 */
  free((void *) image );   
  free_image_double(angles);
  free_image_double(modgrad);
  free_image_char(used);
  free((void *) reg );
  free((void *) mem_p );
  free((void *) output); 
  free((void *) output_2);
  free((void *) output_4);

  /* return the result */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL )
    {
      if( region == NULL ) error("'region' should be a valid image.");
      *reg_img = region->data;
      if( region->xsize > (unsigned int) INT_MAX ||
          region->xsize > (unsigned int) INT_MAX )
        error("region image to big to fit in INT sizes.");
      *reg_x = (int) (region->xsize);
      *reg_y = (int) (region->ysize);

      /* free the 'region' structure.
         we cannot use the function 'free_image_int' because we need to keep
         the memory with the image data to be returned by this function. */
      free( (void *) region );
    }
  if( out->size > (unsigned int) INT_MAX )
    error("too many detections to fit in an INT.");

  /* only the 'ntuple_list' structure must be freed,
  but the 'values' pointer must be keep to return
  as a result. */
  *n_out = (int) (out->size);
  return_value = out->values;
  free( (void *) out );  
  return return_value;
}

