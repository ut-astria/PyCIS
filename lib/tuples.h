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
/*---------------------------- tuples.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef TUPLES_HEADER
#define TUPLES_HEADER

//Define functions
/*----------------------------------------------------------------------------*/
/** 'list of n-tuple' data type

    The i-th component of the j-th n-tuple of an n-tuple list 'ntl'
    is accessed with:
      ntl->values[ i + j * ntl->dim ]
    The dimension of the n-tuple (n) is:
      ntl->dim
    The number of n-tuples in the list is:
      ntl->size
    The maximum number of n-tuples that can be stored in the
    list with the allocated memory at a given time is given by:
      ntl->max_size
 */
typedef struct ntuple_list_s
{
  unsigned int size;
  unsigned int max_size;
  unsigned int dim;
  double * values;
} * ntuple_list;

void free_ntuple_list(ntuple_list in);
ntuple_list new_ntuple_list(unsigned int dim);
void enlarge_ntuple_list(ntuple_list n_tuple);
void add_7tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7 );
void add_10tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7,
                double v8, double v9, double v10);

/*----------------------------------------------------------------------------*/
/** char image data type
    The pixel value at (x,y) is accessed by:
      image->data[ x + y * image->xsize ]
    with x and y integer.
 */
typedef struct image_char_s
{
  unsigned char * data;
  unsigned int xsize,ysize;
} * image_char;

void free_image_char(image_char i);
image_char new_image_char(unsigned int xsize, unsigned int ysize);
image_char new_image_char_ini( unsigned int xsize, unsigned int ysize,
                                      unsigned char fill_value );
 
/*----------------------------------------------------------------------------*/
/** int image data type
    The pixel value at (x,y) is accessed by:
      image->data[ x + y * image->xsize ]
    with x and y integer.
 */
typedef struct image_int_s
{
  int * data;
  unsigned int xsize,ysize;
} * image_int;

image_int new_image_int(unsigned int xsize, unsigned int ysize);
image_int new_image_int_ini( unsigned int xsize, unsigned int ysize,
                                    int fill_value );

/*----------------------------------------------------------------------------*/
/** double image data type
    The pixel value at (x,y) is accessed by:
      image->data[ x + y * image->xsize ]
    with x and y integer.
 */
typedef struct image_double_s
{
  double * data;
  unsigned int xsize,ysize;
} * image_double;

void free_image_double(image_double i);
image_double new_image_double(unsigned int xsize, unsigned int ysize);
image_double new_image_double_ptr( unsigned int xsize,
                                          unsigned int ysize, double * data );

/*----------------------------------------------------------------------------*/
/** char image data type
    The pixel value at (x,y) is accessed by:
      image->data[ x + y * image->xsize ]
    with x and y integer.
 */
typedef struct image3_char_s
{
  unsigned char * data;
  unsigned int xsize,ysize,zsize;
} * image3_char;

void free_image3_char(image3_char i);
image3_char new_image3_char(unsigned int xsize, unsigned int ysize, unsigned int zsize);
image3_char new_image3_char_ini( unsigned int xsize, unsigned int ysize, unsigned int zsize,
                                      unsigned char fill_value );


/*----------------------------------------------------------------------------*/
/** int image data type
    The pixel value at (x,y) is accessed by:
      image->data[ x + y * image->xsize ]
    with x and y integer.
 */
typedef struct image3_int_s
{
  int * data;
  unsigned int xsize,ysize,zsize;
} * image3_int;

image3_int new_image3_int(unsigned int xsize, unsigned int ysize, unsigned int zsize);
image3_int new_image3_int_ini( unsigned int xsize, unsigned int ysize, unsigned int zsize,
                                    int fill_value );


/*----------------------------------------------------------------------------*/
/** double image data type
    The pixel value at (x,y) is accessed by:
      image->data[ x + y * image->xsize ]
    with x and y integer.
 */
typedef struct image3_double_s
{
  double * data;
  unsigned int xsize,ysize,zsize;
} * image3_double;

void free_image3_double(image3_double i);
image3_double new_image3_double(unsigned int xsize, unsigned int ysize, unsigned int zsize);
image3_double new_image3_double_ptr( unsigned int xsize,
                                          unsigned int ysize, unsigned int zsize,  double * data );


//Close header
#endif
