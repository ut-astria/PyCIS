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
/*---------------------------- markov.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef MARKOV_HEADER
#define MARKOV_HEADER

//Define functions
void NFA_matrix(double *output,double p0,double p11,double p01,int N);
void make_markov( double * img, int X, int Y,
                           double ang_th, int n_bins,
                           double * inputv,double inputv_size);
static int isaligned3_markovVORTH(double grads_az,double grads_el,double cprec);
static int isaligned3_markovHORTH(double grads_az,double grads_el,double cprec);
static int isaligned3_markovDORTH(double grads_az,double grads_el,double cprec);
static int isaligned3_markovV(double grads_az,double grads_el,double cprec);
static int isaligned3_markovH(double grads_az,double grads_el,double cprec);
static int isaligned3_markovD(double grads_az,double grads_el,double cprec);
void make_markov3( double * img, int X, int Y, int Z,
                          double ang_th, int n_bins, double * inputv,double inputv_size, 
                          int orth);
                          
//Close header
#endif
