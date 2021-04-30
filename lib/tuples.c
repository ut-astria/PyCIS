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
#include<gsl/gsl_sf_trig.h>
#include<sys/mman.h>

#include "tuples.h"
#include "constants.h"
#include "misc.h"

/*----------------------------------------------------------------------------*/
/*----------------------- 'list of n-tuple' data type ------------------------*/
/*----------------------------------------------------------------------------*/



/*----------------------------------------------------------------------------*/
/** Free memory used in n-tuple 'in'.
 */
void free_ntuple_list(ntuple_list in)
{
  if( in == NULL || in->values == NULL )
    error("free_ntuple_list: invalid n-tuple input.");
  free( (void *) in->values );
  free( (void *) in );
}

/*----------------------------------------------------------------------------*/
/** Create an n-tuple list and allocate memory for one element.
    @param dim the dimension (n) of the n-tuple.
 */
ntuple_list new_ntuple_list(unsigned int dim)
{
  ntuple_list n_tuple;

  /* check parameters */
  if( dim == 0 ) error("new_ntuple_list: 'dim' must be positive.");

  /* get memory for list structure */
  n_tuple = (ntuple_list) malloc( sizeof(struct ntuple_list_s) );
  if( n_tuple == NULL ) error("not enough memory.");

  /* initialize list */
  n_tuple->size = 0;
  n_tuple->max_size = 1;
  n_tuple->dim = dim;

  /* get memory for tuples */
  n_tuple->values = (double *) malloc( dim*n_tuple->max_size * sizeof(double) );
  if( n_tuple->values == NULL ) error("not enough memory.");

  return n_tuple;
}

/*----------------------------------------------------------------------------*/
/** Enlarge the allocated memory of an n-tuple list.
 */
void enlarge_ntuple_list(ntuple_list n_tuple)
{
  /* check parameters */
  if( n_tuple == NULL || n_tuple->values == NULL || n_tuple->max_size == 0 )
    error("enlarge_ntuple_list: invalid n-tuple.");

  /* duplicate number of tuples */
  n_tuple->max_size *= 2;

  /* realloc memory */
  n_tuple->values = (double *) realloc( (void *) n_tuple->values,
                      n_tuple->dim * n_tuple->max_size * sizeof(double) );
  if( n_tuple->values == NULL ) error("not enough memory.");
}

/*----------------------------------------------------------------------------*/
/** Add a 7-tuple to an n-tuple list.
 */
void add_7tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7 )
{
  /* check parameters */
  if( out == NULL ) error("add_7tuple: invalid n-tuple input.");
  if( out->dim != 7 ) error("add_7tuple: the n-tuple must be a 7-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_7tuple: invalid n-tuple input.");

  /* add new 7-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  out->values[ out->size * out->dim + 4 ] = v5;
  out->values[ out->size * out->dim + 5 ] = v6;
  out->values[ out->size * out->dim + 6 ] = v7;

  /* update number of tuples counter */
  out->size++;
}

/*----------------------------------------------------------------------------*/
/** Add a 10-tuple to an n-tuple list. (used for 3D line output)
 */
void add_10tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7,
                double v8, double v9, double v10)
{
  /* check parameters */
  if( out == NULL ) error("add_10tuple: invalid n-tuple input.");
  if( out->dim != 10 ) error("add_10tuple: the n-tuple must be a 10-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_10tuple: invalid n-tuple input.");

  /* add new 7-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  out->values[ out->size * out->dim + 4 ] = v5;
  out->values[ out->size * out->dim + 5 ] = v6;
  out->values[ out->size * out->dim + 6 ] = v7;
  out->values[ out->size * out->dim + 7 ] = v8;
  out->values[ out->size * out->dim + 8 ] = v9;
  out->values[ out->size * out->dim + 9 ] = v10;
  
  /* update number of tuples counter */
  out->size++;
}


/*----------------------------------------------------------------------------*/
/*----------------------------- Image Data Types -----------------------------*/
/*----------------------------------------------------------------------------*/



/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
void free_image_char(image_char i)
{
  if( i == NULL || i->data == NULL )
    error("free_image_char: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize'.
 */
image_char new_image_char(unsigned int xsize, unsigned int ysize)
{
  image_char image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_char: invalid image size.");

  /* get memory */
  image = (image_char) malloc( sizeof(struct image_char_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (unsigned char *) calloc( (size_t) (xsize*ysize),
                                          sizeof(unsigned char) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
image_char new_image_char_ini( unsigned int xsize, unsigned int ysize,
                                      unsigned char fill_value )
{
  image_char image = new_image_char(xsize,ysize); /* create image */
  unsigned int N = xsize*ysize;
  unsigned int i;

  /* check parameters */
  if( image == NULL || image->data == NULL )
    error("new_image_char_ini: invalid image.");

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize'.
 */
image_int new_image_int(unsigned int xsize, unsigned int ysize)
{
  image_int image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_int: invalid image size.");

  /* get memory */
  image = (image_int) malloc( sizeof(struct image_int_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (int *) calloc( (size_t) (xsize*ysize), sizeof(int) );
  if( image->data == NULL ) error("not enough memory.");

  /* set imagsavelines=zeros(size(lines,1),5);
for xx=1:size(lines,1)
    savelines(xx,1:4)=lines(xx,1:4);
    savelines(xx,5)=angleline(xx);
ende size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
image_int new_image_int_ini( unsigned int xsize, unsigned int ysize,
                                    int fill_value )
{
  image_int image = new_image_int(xsize,ysize); /* create image */
  unsigned int N = xsize*ysize;
  unsigned int i;

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}



/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
void free_image_double(image_double i)
{
  if( i == NULL || i->data == NULL )
    error("free_image_double: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
image_double new_image_double(unsigned int xsize, unsigned int ysize)
{
  image_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_double: invalid image size.");

  /* get memory */
  image = (image_double) malloc( sizeof(struct image_double_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (double *) calloc( (size_t) (xsize*ysize), sizeof(double) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
    with the data pointed by 'data'.
 */
image_double new_image_double_ptr( unsigned int xsize,
                                          unsigned int ysize, double * data )
{
  image_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 )
    error("new_image_double_ptr: invalid image size.");
  if( data == NULL ) error("new_image_double_ptr: NULL data pointer.");

  /* get memory */
  image = (image_double) malloc( sizeof(struct image_double_s) );
  if( image == NULL ) error("not enough memory.");

  /* set image */
  image->xsize = xsize;
  image->ysize = ysize;
  
  image->data = data;

  return image;
}

/*----------------------------------------------------------------------------*/
/*-------------------------- 3D Image Data Types -----------------------------*/
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
void free_image3_char(image3_char i)
{
  if( i == NULL || i->data == NULL )
    error("free_image3_char: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize'.
 */
image3_char new_image3_char(unsigned int xsize, unsigned int ysize, unsigned int zsize)
{
  image3_char image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0) error("new_image3_char: invalid image size.");

  /* get memory */
  image = (image3_char) malloc( sizeof(struct image3_char_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (unsigned char *) calloc( (size_t) (xsize*ysize*zsize),
                                          sizeof(unsigned char) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;
  image->zsize = zsize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
image3_char new_image3_char_ini( unsigned int xsize, unsigned int ysize, unsigned int zsize,
                                      unsigned char fill_value )
{
  image3_char image = new_image3_char(xsize,ysize,zsize); /* create image */
  unsigned int N = xsize*ysize*zsize;
  unsigned int i;

  /* check parameters */
  if( image == NULL || image->data == NULL )
    error("new_image3_char_ini: invalid image.");

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize'.
 */
image3_int new_image3_int(unsigned int xsize, unsigned int ysize, unsigned int zsize)
{
  image3_int image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0 ) error("new_image3_int: invalid image size.");

  /* get memory */
  image = (image3_int) malloc( sizeof(struct image3_int_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (int *) calloc( (size_t) (xsize*ysize*zsize), sizeof(int) );
  if( image->data == NULL ) error("not enough memory.");

  image->xsize = xsize;
  image->ysize = ysize;
  image->zsize = zsize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
image3_int new_image3_int_ini( unsigned int xsize, unsigned int ysize, unsigned int zsize,
                                    int fill_value )
{
  image3_int image = new_image3_int(xsize,ysize,zsize); /* create image */
  unsigned int N = xsize*ysize*zsize;
  unsigned int i;

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}


/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
void free_image3_double(image3_double i)
{
  if( i == NULL || i->data == NULL )
    error("free_image3_double: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
image3_double new_image3_double(unsigned int xsize, unsigned int ysize, unsigned int zsize)
{
  image3_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0) error("new_image3_double: invalid image size.");

  /* get memory */
  image = (image3_double) malloc( sizeof(struct image3_double_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (double *) calloc( (size_t) (xsize*ysize*zsize), sizeof(double) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;
  image->zsize = zsize;
  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
    with the data pointed by 'data'.
 */
image3_double new_image3_double_ptr( unsigned int xsize,
                                          unsigned int ysize, unsigned int zsize,  double * data )
{
  image3_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 || zsize == 0 )
    error("new_image3_double_ptr: invalid image size.");
  if( data == NULL ) error("new_image3_double_ptr: NULL data pointer.");

  /* get memory */
  image = (image3_double) malloc( sizeof(struct image3_double_s) );
  if( image == NULL ) error("not enough memory.");

  /* set image */
  image->xsize = xsize;
  image->ysize = ysize;
  image->zsize = zsize;
  
  image->data = data;

  return image;
}

