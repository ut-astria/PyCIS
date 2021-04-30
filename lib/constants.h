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
/*---------------------------- constants.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef CONSTANTS_HEADER
#define CONSTANTS_HEADER

/*--------------------------------------------------------------------------
  Define constants 
*/
/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */

/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

#ifndef RADIANS_TO_DEGREES
#define RADIANS_TO_DEGREES (180.0/M_PI)
#endif /*!RADIANS_TO_DEGREES*/

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** Label for pixels with undefined gradient. */
#ifndef NOTDEF
#define NOTDEF -1024.0
#endif
/** 3/2 pi */
#ifndef M_3_2_PI
#define M_3_2_PI 4.71238898038
#endif
/** 2 pi */
#ifndef M_2__PI
#define M_2__PI  6.28318530718
#endif
/** Label for pixels not used in yet. */
#ifndef NOTUSED
#define NOTUSED 0
#endif
/** Label for pixels already used in detection. */
#ifndef USED
#define USED    1
#endif 

/*----------------------------------------------------------------------------*/
/** Doubles relative error factor
 */
#ifndef RELATIVE_ERROR_FACTOR
#define RELATIVE_ERROR_FACTOR 100.0
#endif

//Close header
#endif /* !LSD_HEADER */