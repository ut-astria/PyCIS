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
/*---------------------------- gradient.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef GRADIENT_HEADER
#define GRADIENT_HEADER

#include "tuples.h"
struct coorlist; //incomplete forward declaration 
struct coorlist3; //incomplete forward declaration 


//Define Functions
image_double ll_angle( image_double in,
                              struct coorlist ** list_p, void ** mem_p,
                              image_double * modgrad, unsigned int n_bins,double alpha);

/*----------------------------------------------------------------------------*/
/* Storage structure for 3D gradients. grads->az, grads->el,
 * instatiate as struct grads newgrad*/
typedef struct grads_s {
  image3_double az;
  image3_double el;
} * grads;

grads new_grads(unsigned int xsize, unsigned int ysize, unsigned int zsize);
void free_grads(grads i);
grads ll_angle3( image3_double in,
                        struct coorlist3 ** list_p, void ** mem_p,
                        image3_double * modgrad, 
                        unsigned int n_bins,double alpha);

int isaligned( int x, int y, image_double angles, double theta,  double prec );
int isaligned3(double grads_az,double grads_el,double theta_az,double theta_el,double prec);
int isaligned3ORTH(double grads_az,double grads_el,double theta_az,double theta_el,double prec);

//Close header
#endif /* !LSD_HEADER */