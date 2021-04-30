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
/*---------------------------- regions3.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef REGIONS3_HEADER
#define REGIONS3_HEADER

#include "misc.h" //required for angles3
#include "tuples.h"
#include "gradient.h" //regquired fro grads
#include "rectangles3.h"
struct point3; //incomplete forward declaration from misc.h

//Define functions
angles3 get_theta3( struct point3 * reg, int reg_size, double x, double y, double z,
                        image3_double modgrad, angles3 reg_angle, double prec, int orth );
void region2rect3( struct point3 * reg, int reg_size,
                        image3_double modgrad, angles3 reg_angle,
                        double prec, double p, struct rect3 * rec , int orth );
void region3_grow(int x, int y,int z, grads angles, 
                        struct point3 * reg,
                        int * reg_size, angles3 * reg_angle, 
                        image3_char used,double prec ,int NOUT);
void region3_growORTH(int x, int y,int z, 
                        image3_double modgrad, grads angles, 
                        struct point3 * reg, int * reg_size, 
                        angles3 * reg_angle,  angles3 * lstheta, 
                            image3_char used,double prec ,int NOUT);
double rect3_improve_update(struct rect3  r, grads angles,double logNT,int Nnfa,
                        double* mnfa, double* mnfap, int minsize,
                        double* mnfa_2,double* mnfap_2, int minsize2,
                        double* mnfa_4,double* mnfap_4, int minsize4,
                        double p1check, double p2check,
                        struct rect3 * rec,double log_nfa,int orth);
double rect3_improve( struct rect3 * rec, grads angles,
                        double logNT, double log_eps,
                        double* mnfa,double* mnfa_2,double* mnfa_4,
                        double*mnfap,double*mnfap_2,double*mnfap_4,
                        int Nnfa,int minsize, int minsize2,int minsize4,int orth);
int reduce_region3_radius( struct point3 * reg, int * reg_size,
                        image3_double modgrad, angles3 reg_angle,
                        double prec, double p, struct rect3 * rec,
                        image3_char used, grads angles,
                        double density_th , int orth);
int refine3( struct point3 * reg, int * reg_size, image3_double modgrad,
                        angles3 reg_angle, double prec, double p, struct rect3 * rec,
                        image3_char used, grads angles,
                        double density_th , int NOUT, int orth);

                                

//Close header
#endif
