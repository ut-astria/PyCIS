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
/*---------------------------- misc.h --------------------------------*/
/*----------------------------------------------------------------------------*/

//Open header
#ifndef MISC_HEADER
#define MISC_HEADER

//Define functions
int max1(int x, int y);
int min1(int x, int y);
double max2(double x, double y);
double min2(double x, double y);
/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist
{
  int x,y;
  struct coorlist * next;
};
struct coorlist3
{
    int x,y,z;
    struct coorlist3 * next;
};
/*----------------------------------------------------------------------------*/
/** A point (or pixel).
 */
struct point {int x,y;};
struct point3 {int x,y,z;};

void error(char * msg);
int double_equal(double a, double b);

double dist(double x1, double y1, double x2, double y2);
double dist3(double x1, double y1, double z1, double x2, double y2, double z2);
double line_angle(double x1, double y1, double x2, double y2);

/*----------------------------------------------------------------------------*/
/*Orientation of a line in sphereical coordinates
 * (az,el) represents the principal axis of a line, while
 * (az3,el3) and (az2,el2) represent the minor and intermediate axes, 
 * for consideration of planar events.
*/
typedef struct angles3_s
{
  double az,el;  
  double az2,el2;
  double az3,el3;
} * angles3;

angles3 new_angles3(double az, double el);
void free_angles3(angles3 i);
angles3 line_angle3(double x1, double y1, double z1, 
        double x2, double y2, double z2);
double angle_diff(double a, double b);
double angle_diff_signed(double a, double b);

//Close header
#endif
