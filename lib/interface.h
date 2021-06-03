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
/*---------------------------- interface.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef INTERFACE_HEADER
#define INTERFACE_HEADER

//Define Functions

double * lsd(int * n_out, 
            double * img, int X, int Y,
            double *inputv, double inputv_size);

double * lsd3(int * n_out,
            double * img, int X, int Y, int Z, 
            double *inputv, double inputv_size,double * inputvorth);

double * lsd3b(int * n_out,
            double * img, int X, int Y, int Z, 
            double *inputv, double inputv_size,double * inputvorth);

double * lsd3center(int * n_out, 
            double * img, int X, int Y, int Z,
            double * img0, int X0, int Y0, int Z0,
            double *inputv, double inputv_size,double * inputvorth);

double * lsd3centerb(int * n_out, 
            double * img, int X, int Y, int Z,
            double * img0, int X0, int Y0, int Z0,
            double *inputv, double inputv_size,double * inputvorth);

double * lsdM(int * n_out, 
            double * img, int X, int Y,
            double * img0, int X0, int Y0,
            double *inputv, double inputv_size);

double * lsd3M(int * n_out, 
            double * img, int X, int Y, int Z,
            double * img0, int X0, int Y0, int Z0,
            double *inputv, double inputv_size,double * inputvorth);

//Close header
#endif /* !LSD_HEADER */