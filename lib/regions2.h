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
/*---------------------------- regions2.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef REGIONS2_HEADER
#define REGIONS2_HEADER

#include "tuples.h"
#include "rectangles2.h"
struct point; //incomplete forward declaration from misc;


//Define functions
double get_theta( struct point * reg, int reg_size, double x, double y,
                        image_double modgrad, double reg_angle, double prec );
void region2rect( struct point * reg, int reg_size,
                        image_double modgrad, double reg_angle,
                        double prec, double p, struct rect * rec );
void region_grow( int x, int y, image_double angles, struct point * reg,
                        int * reg_size, double * reg_angle, image_char used,
                        double prec );
void region_growORTH( int x, int y, image_double angles, 
                        struct point * reg, int * reg_size, 
                        double * reg_angle, double * lstheta, 
                        image_char used, double prec);

double rect_improve( struct rect * rec, image_double angles,
                        double logNT, double log_eps,
                        double* mnfa,double* mnfa_2,double* mnfa_4,
                        int Nnfa,int minsize, int minsize2,int minsize4 );
double rect_improve_update(struct rect  r, image_double angles,double logNT,int Nnfa,
                        double* mnfa, double* mnfap, int minsize,
                        double* mnfa_2,double* mnfap_2, int minsize2,
                        double* mnfa_4,double* mnfap_4, int minsize4,
                        double p1check, double p2check,
                        struct rect * rec,double log_nfa,int orth);
double rect_improveORTH( struct rect * rec, image_double angles,
                        double logNT, double log_eps,
                        double* mnfa,double* mnfa_2,double* mnfa_4,
                        double*mnfap,double*mnfap_2,double*mnfap_4,
                        int Nnfa,int minsize, int minsize2,int minsize4, int orth );
int reduce_region_radius( struct point * reg, int * reg_size,
                        image_double modgrad, double reg_angle,
                        double prec, double p, struct rect * rec,
                        image_char used, image_double angles,
                        double density_th );
int refine( struct point * reg, int * reg_size, image_double modgrad,
                        double reg_angle, double prec, double p, struct rect * rec,
                        image_char used, image_double angles, double density_th );
int refineORTH( struct point * reg, int * reg_size, image_double modgrad,
                   double reg_angle, double prec, double p, struct rect * rec,
                   image_char used, image_double angles, double density_th ,int orth);







//Close header
#endif
