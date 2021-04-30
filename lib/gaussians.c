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
#include <gsl/gsl_sf_trig.h>
#include <sys/mman.h>

#include "gaussians.h"
#include "constants.h"
#include "misc.h"
#include "tuples.h"
/*----------------------------------------------------------------------------*/
/*----------------------------- Gaussian filter ------------------------------*/
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/** Compute a Gaussian kernel of length 'kernel->dim',
    standard deviation 'sigma', and centered at value 'mean'.

    For example, if mean=0.5, the Gaussian will be centered
    in the middle point between values 'kernel->values[0]'
    and 'kernel->values[1]'.

    UPDATE: Used in both 2D and 3D by seperability 
 */
static void gaussian_kernel(ntuple_list kernel, double sigma, double mean)
{
  double sum = 0.0;
  double val;
  unsigned int i;

  /* check parameters */
  if( kernel == NULL || kernel->values == NULL )
    error("gaussian_kernel: invalid n-tuple 'kernel'.");
  if( sigma <= 0.0 ) error("gaussian_kernel: 'sigma' must be positive.");

  /* compute Gaussian kernel */
  if( kernel->max_size < 1 ) enlarge_ntuple_list(kernel);
  kernel->size = 1;
  for(i=0;i<kernel->dim;i++)
    {
      val = ( (double) i - mean ) / sigma;
      kernel->values[i] = exp( -0.5 * val * val );
      sum += kernel->values[i];
    }

  /* normalization */
  if( sum >= 0.0 ) for(i=0;i<kernel->dim;i++) kernel->values[i] /= sum;
}

/*----------------------------------------------------------------------------*/
/** Scale the input image 'in' by a factor 'scale' by Gaussian sub-sampling.

    For example, scale=0.8 will give a result at 80% of the original size.

    The image is convolved with a Gaussian kernel
    @f[
        G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
    @f]
    before the sub-sampling to prevent cosasing.

    The standard deviation sigma given by:
    -  sigma = sigma_scale / scale,   if scale <  1.0
    -  sigma = sigma_scale,           if scale >= 1.0

    To be able to sub-sample at non-integer steps, some interpolation
    is needed. In this implementation, the interpolation is done by
    the Gaussian kernel, so both operations (filtering and sampling)
    are done at the same time. The Gaussian kernel is computed
    centered on the coordinates of the required sample. In this way,
    when applied, it gives directly the result of convolving the image
    with the kernel and interpolated to that particular position.

    A fast algorithm is done using the separability of the Gaussian
    kernel. Applying the 2D Gaussian kernel is equivalent to applying
    first a horizontal 1D Gaussian kernel and then a vertical 1D
    Gaussian kernel (or the other way round). The reason is that
    @f[
        G(x,y) = G(x) * G(y)
    @f]
    where 
    @f[
        G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}.
    @f]
    The algorithm first applies a combined Gaussian kernel and sampling
    in the x axis, and then the combined Gaussian kernel and sampling
    in the y axis.
 */
image_double gaussian_sampler( image_double in, double scale,
                                      double sigma_scale )
{
  image_double aux,out;
  ntuple_list kernel;
  unsigned int N,M,h,n,x,y,i;
  int xc,yc,j,double_x_size,double_y_size;
  double sigma,xx,yy,sum,prec;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 )
    error("gaussian_sampler: invalid image.");
  if( scale <= 0.0 ) error("gaussian_sampler: 'scale' must be positive.");
  if( sigma_scale <= 0.0 )
    error("gaussian_sampler: 'sigma_scale' must be positive.");

  /* compute new image size and get memory for images */
  if( in->xsize * scale > (double) UINT_MAX ||
      in->ysize * scale > (double) UINT_MAX )
    error("gaussian_sampler: the output image size exceeds the handled size.");
  N = (unsigned int) ceil( in->xsize * scale );
  M = (unsigned int) ceil( in->ysize * scale );
  aux = new_image_double(N,in->ysize);
  out = new_image_double(N,M);

  /* sigma, kernel size and memory for the kernel */
  sigma = scale < 1.0 ? sigma_scale / scale : sigma_scale;
  /*
     The size of the kernel is selected to guarantee that the
     the first discarded term is at least 10^prec times smaller
     than the central value. For that, h should be larger than x, with
       e^(-x^2/2sigma^2) = 1/10^prec.
     Then,
       x = sigma * sqrt( 2 * prec * ln(10) ).
   */
  prec = 3.0;
  h = (unsigned int) ceil( sigma * sqrt( 2.0 * prec * log(10.0) ) );
  n = 1+2*h; /* kernel size */
  kernel = new_ntuple_list(n);

  /* auxiliary double image size variables */
  double_x_size = (int) (2 * in->xsize);
  double_y_size = (int) (2 * in->ysize);

  /* First subsampling: x axis */
  for(x=0;x<aux->xsize;x++)
    {
      /*
         x   is the coordinate in the new image.
         xx  is the corresponding x-value in the original size image.
         xc  is the integer value, the pixel coordinate of xx.
       */
      xx = (double) x / scale;
      /* coordinate (0.0,0.0) is in the center of pixel (0,0),
         so the pixel with xc=0 get the values of xx from -0.5 to 0.5 */
      xc = (int) floor( xx + 0.5 );
      gaussian_kernel( kernel, sigma, (double) h + xx - (double) xc );
      /* the kernel must be computed for each x because the fine
         offset xx-xc is different in each case */

      for(y=0;y<aux->ysize;y++)
        {
          sum = 0.0;
          for(i=0;i<kernel->dim;i++)
            {
              j = xc - h + i;

              /* symmetry boundary condition */
              while( j < 0 ) j += double_x_size;
              while( j >= double_x_size ) j -= double_x_size;
              if( j >= (int) in->xsize ) j = double_x_size-1-j;

              sum += in->data[ j + y * in->xsize ] * kernel->values[i];
            }
          aux->data[ x + y * aux->xsize ] = sum;
        }
    }

  /* Second subsampling: y axis */
  for(y=0;y<out->ysize;y++)
    {
      /*
         y   is the coordinate in the new image.
         yy  is the corresponding x-value in the original size image.
         yc  is the integer value, the pixel coordinate of xx.
       */
      yy = (double) y / scale;
      /* coordinate (0.0,0.0) is in the center of pixel (0,0),
         so the pixel with yc=0 get the values of yy from -0.5 to 0.5 */
      yc = (int) floor( yy + 0.5 );
      gaussian_kernel( kernel, sigma, (double) h + yy - (double) yc );
      /* the kernel must be computed for each y because the fine
         offset yy-yc is different in each case */

      for(x=0;x<out->xsize;x++)
        {
          sum = 0.0;
          for(i=0;i<kernel->dim;i++)
            {
              j = yc - h + i;

              /* symmetry boundary condition */
              while( j < 0 ) j += double_y_size;
              while( j >= double_y_size ) j -= double_y_size;
              if( j >= (int) in->ysize ) j = double_y_size-1-j;

              sum += aux->data[ x + j * aux->xsize ] * kernel->values[i];
            }
          out->data[ x + y * out->xsize ] = sum;
        }
    }

  /* free memory */
  free_ntuple_list(kernel);
  free_image_double(aux);

  return out;
}

/*----------------------------------------------------------------------------*/
/** Scale the input image 'in' by a factor 'scale' using Gaussian sub-sampling.
    3D extension of the fast 2D alg. (see 'gaussian_sampler')
 */
image3_double gaussian3_sampler( image3_double in, double scale,
                                      double sigma_scale )
{
  image3_double aux,aux2,out;
  ntuple_list kernel;
  unsigned int N,M,O,h,n,x,y,z,i;
  int xc,yc,zc,j,double_x_size,double_y_size,double_z_size;
  double sigma,xx,yy,zz,sum,prec;

  printf("\t Inside Gauss: scale %.2f, sigma %.2f \n",scale,sigma_scale); fflush(stdout);
  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0|| in->zsize==0 )
    error("gaussian3_sampler: invalid image.");
  if( scale <= 0.0 ) error("gaussian_sampler: 'scale' must be positive.");
  if( sigma_scale <= 0.0 )
    error("gaussian_sampler: 'sigma_scale' must be positive.");

  /* compute new image size and get memory for images */
  if( in->xsize * scale > (double) UINT_MAX ||
      in->ysize * scale > (double) UINT_MAX ||
      in->zsize * scale > (double) UINT_MAX)
    error("gaussian3_sampler: the output image size exceeds the handled size.");
  O = (unsigned int) ceil( in->zsize * scale );
  N = (unsigned int) ceil( in->xsize * scale );
  M = (unsigned int) ceil( in->ysize * scale );
  unsigned int imgSize = (unsigned int) M*N*O;
  /* instantiating auxilliary matrices*/
  aux = new_image3_double(N,in->ysize,in->zsize);
  aux2= new_image3_double(N,M,in->zsize);
  out = new_image3_double(N,M,O);

  /* sigma, kernel size and memory for the kernel */
  sigma = scale < 1.0 ? sigma_scale / scale : sigma_scale;
  /*
     The size of the kernel is selected to guarantee that the
     the first discarded term is at least 10^prec times smaller
     than the central value. For that, h should be larger than x, with
       e^(-x^2/2sigma^2) = 1/10^prec.
     Then,
       x = sigma * sqrt( 2 * prec * ln(10) ).
   */
  prec = 3.0;
  h = (unsigned int) ceil( sigma * sqrt( 2.0 * prec * log(10.0) ) );
  n = 1+2*h; /* kernel size */
  kernel = new_ntuple_list(n);


  /* auxiliary double image size variables */
  double_x_size = (int) (2 * in->xsize);
  double_y_size = (int) (2 * in->ysize);
  double_z_size = (int) (2 * in->zsize);

  for(x=0;x<aux->xsize;x++){
    xx = (double) x / scale;
    xc = (int) floor( xx + 0.5 );
    gaussian_kernel( kernel, sigma, (double) h + xx - (double) xc );
    for(y=0;y<aux->ysize;y++)
      for (z=0;z<aux->zsize;z++) {
        sum = 0.0;
        for(i=0;i<kernel->dim;i++) {
          j = xc - h + i;
          while( j < 0 ) j += double_x_size;
          while( j >= double_x_size ) j -= double_x_size;
          if( j >= (int) in->xsize ) j = double_x_size-1-j;
          sum += in->data[ z + (j + y * in->xsize) * in->zsize ] * kernel->values[i];
          }
        aux->data[ z + (x + y * aux->xsize) * aux->zsize ] = sum;
  }}
  for(y=0;y<aux2->ysize;y++)  {
    yy = (double) y / scale;
    yc = (int) floor( yy + 0.5 );
    gaussian_kernel( kernel, sigma, (double) h + yy - (double) yc );
    for(x=0;x<aux2->xsize;x++)
	    for(z=0;z<aux2->zsize;z++){
        sum = 0.0;
        for(i=0;i<kernel->dim;i++) {
            j = yc - h + i;
            while( j < 0 ) j += double_y_size;
            while( j >= double_y_size ) j -= double_y_size;
            if( j >= (int) in->ysize ) j = double_y_size-1-j;
	          sum += aux->data[ z + (x + j * aux->xsize) * aux->zsize] * kernel->values[i];
            }
          aux2->data[ z + (x + y * aux2->ysize) * aux2->zsize ] = sum;
  }}
  for(z=0;z<out->zsize;z++) {
    zz = (double) z / scale;
    zc = (int) floor( zz + 0.5 );
    gaussian_kernel( kernel, sigma, (double) h + zz - (double) zc );
    for(x=0;x<out->xsize;x++)
      for(y=0;y<out->ysize;y++)  {
        sum = 0.0;
        for(i=0;i<kernel->dim;i++) {
            j = zc - h + i;
            while( j < 0 ) j += double_z_size;
            while( j >= double_z_size ) j -= double_z_size;
            if( j >= (int) in->zsize ) j = double_z_size-1-j;
            sum += aux2->data[ j + (x + y * aux2->xsize)*aux2->zsize]*kernel->values[i];
          }
        out->data[ z + (x + y * out->xsize) * out->zsize ] = sum;
  }}

  /* free memory */
  free_ntuple_list(kernel);
  free_image3_double(aux);
  free_image3_double(aux2);
  //free((void *) aux);
  //free((void *) aux2);
  return out;
}
