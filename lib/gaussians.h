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
/*---------------------------- gaussainss.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef GAUSSIANS_HEADER
#define GAUSSIANS_HEADER

#include "tuples.h"

//Define functions
//NOTE: Require top-level import of tuples.h

static void gaussian_kernel(ntuple_list kernel, double sigma, double mean);

image_double gaussian_sampler( image_double in, double scale,
                                      double sigma_scale );

image3_double gaussian3_sampler( image3_double in, double scale,
                                      double sigma_scale );

//Close header
#endif
