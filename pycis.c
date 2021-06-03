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

*NOTICE: This program is modified from the source code of LSDSAR:
*"LSDSAR, a Markovian a contrario framework for line segment detection in SAR images"
*by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. 
*(Pattern Recognition, 2019).
*https://doi.org/10.1016/j.patcog.2019.107034
*Date of Modification: April 30, 2021


*NOTICE: This program is released under GNU Affero General Public License
*and any conditions added under section 7 in the link:
*https://www.gnu.org/licenses/agpl-3.0.en.html

Copyright (c) 2021 Benjamin Feuge-Miller <benjamin.g.miller@utexas.edu>

This program is free software: you can redistribute it and/or modify 
 it under the terms of the GNU General Public License as published 
 by the Free Software Foundation, either version 3 of the License, 
 or (at your option) any later version.

This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.
 
You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------------------

Modified functions under appropriate headers: 

gaussians.h -     gaussian downsampling for antialising
  image3_double gaussian3_sampler( image3_double in, double scale, double sigma_scale )
    [extension to 3d]

gradient.h -      compute gradient magnitude/orientation, and alignment checks
  grads new_grads(unsigned int xsize, unsigned int ysize, unsigned int zsize)
    [structure for polar orientations]
  void free_grads(grads i)
  grads ll_angle3( image3_double in, struct coorlist3 ** list_p, void ** mem_p,
                   image3_double * modgrad,  unsigned int n_bins,double alpha)
    [extension to 3d with omp parallelism]

interface.h -     pipeline helpers for pycis.c
  double * lsd(int * n_out, double * img, int X, int Y, double *inputv, double inputv_size);
  double * lsd3(int * n_out, double * img, int X, int Y, int Z, double *inputv, double inputv_size,double * inputvorth);
  double * lsd3center(int * n_out, double * img, int X, int Y, int Z, double * img0, int X0, int Y0, int Z0,
                      double *inputv, double inputv_size,double * inputvorth);
  double * lsdM(int * n_out, double * img, int X, int Y, double * img0, int X0, int Y0, double *inputv, double inputv_size);
  double * lsd3M(int * n_out, double * img, int X, int Y, int Z, double * img0, int X0, int Y0, int Z0,
                 double *inputv, double inputv_size,double * inputvorth);
    [pipeline helper functions calling lsd(2/3).h]

lsd3.h -        main lsd pipelines
  double * LineSegmentDetection3( int * n_out, double * img, int X, int Y, int Z, double ang_th, double log_eps, 
                                  double density_th, int n_bins, int ** reg_img, int * reg_x, int * reg_y, int * reg_z, 
                                  double * inputv, double inputv_size, double * inputvorth)
    [extension to 3d with extended variables for nfa estimation ]
  double * LineSegmentDetection3Center( int * n_out, double * img, int X, int Y, int Z, double * img0, int X0, int Y0, 
                                        int Z0, double ang_th, double log_eps, double density_th, int n_bins, 
                                        int ** reg_img, int * reg_x, int * reg_y, int * reg_z, double * inputv, 
                                        double inputv_size, double * inputvorth)
    [variation using prior lines to run alternative region growing/ nfa estimation algorithms]

markov.h -        estimate markov kernels and build transition matrices
  void make_markov( double * img, int X, int Y, double ang_th, int n_bins, double * inputv,double inputv_size)
  void make_markov3( double * img, int X, int Y, int Z, double ang_th, int n_bins, 
                     double * inputv,double inputv_size, int orth)
    [compute image statistics for markov kernel estimation]
  static int isaligned3_markovV(double grads_az,double grads_el,double cprec)
    [see also markovH, markovD, and markov<X>ORTH functions]
    [helper alignment functions for faster parallel/orthogonal evaluations]

misc.h -          lists, angle functions
  double dist3(double x1, double y1, double z1, double x2, double y2, double z2)
    [pointwise cartesian distance in 3d]
  angles3 new_angles3(double az, double el)
    [structure for storing polar angles]
  void free_angles3(angles3 i)
  angles3 line_angle3(double x1, double y1, double z1, double x2, double y2, double z2)
    [compute polar angles]

nfa.h -           estimate markov or negative binomial approximation tail probabiltiy
  double nfaORTH(int n, int k, double pp, double logNT, double *mnfa, int N)
    [negative binomial approximation, for very long markov chains in 3d orthogonality checks]

rectangles2.h- build rectangle objects and iterator to call nfa
  int ri_end(rect_iter * i)
    [update for iteration in projected space]
  void up_all(rect_iter * i)
    [project points between principal and cartesian space using polar transform]
  void ri_inc(rect_iter * i)
    [integer pixel iteration in projected space]
  rect_iter * ri_ini(struct rect * r)
    [update jacobian for proper iteration]

rectangles3.h- build rectangle objects and iterator to call nfa
  [see rectangles2.h for ri3_<function> functions extended to 3d]
  double rect3_nfa(struct rect3 * rec, grads angles, double logNT,double *image,int N,int minreg)
    [in-line alignment checks and orientation comparison for edge detection in 3d]
  double rect3_nfaORTH(struct rect3 * rec, grads angles, double logNT,double *image,double *pset, int N,int minreg)
    [update rect3_nfa for centerline detection with markov estimation for very large point volumes]
    
regions3.h -  build and improve pixel regions for estimating rectangles
  angles3 get_theta3( struct point3 * reg, int reg_size, double x, double y, double z,
                      image3_double modgrad, angles3 reg_angle, double prec, int orth )
    [compute and store polar orientations of principal directions, using gsl eigensovler for speed]
  void region2rect3( struct point3 * reg, int reg_size, image3_double modgrad, angles3 reg_angle,
                     double prec, double p, struct rect3 * rec , int orth )
    [extension to 3d with improved jacobian handling]
  void region3_grow(int x, int y,int z, grads angles, struct point3 * reg, int * reg_size, 
                    angles3 * reg_angle, image3_char used,double prec ,int NOUT)
    [edge-surface-feature growing in 3d with parallel alignment between pixels]
  void region3_growORTH(int x, int y,int z, image3_double modgrad, grads angles, struct point3 * reg, int * reg_size, 
                        angles3 * reg_angle,  angles3 * lstheta, image3_char used,double prec ,int NOUT)
    [center-volume-feature growing in 3d with orthogonal alignment between pixels and an a-priori principal axis guess]
    [TODO: orientation if fixed due to linearity constraint, need to update with projection to ra/dec after edge detection]
  double rect3_improve_update(struct rect3  r, grads angles,double logNT,int Nnfa, double* mnfa, double* mnfap, int minsize,
                                 double* mnfa_2,double* mnfap_2, int minsize2, double* mnfa_4,double* mnfap_4, int minsize4,
                                 double p1check, double p2check, struct rect3 * rec,double log_nfa,int orth)
    [helper function for edge-detection/ center-line-detetection nfa evaluation on different angular tolerances]
  double rect3_improve( struct rect3 * rec, grads angles, double logNT, double log_eps, double* mnfa,double* mnfa_2,double* mnfa_4,
                            double*mnfap,double*mnfap_2,double*mnfap_4, int Nnfa,int minsize, int minsize2,int minsize4,int orth) 
    [extension to 3d with geometric width sequences, assuming rectangular prism features]
  int reduce_region3_radius( struct point3 * reg, int * reg_size, image3_double modgrad, angles3 reg_angle, double prec, double p, 
                                 struct rect3 * rec, image3_char used, grads angles, double density_th , int orth)
    [extension to 3d assuming rectangular prism features]
  int refine3( struct point3 * reg, int * reg_size, image3_double modgrad, angles3 reg_angle, double prec, double p, 
               struct rect3 * rec,  image3_char used, grads angles, double density_th , int NOUT, int orth)
    [extension to 3d]

tuples.h -        construction of line tuples and image structures 
  void add_10tuple( ntuple_list out, double v1, double v2, double v3, double v4, double v5, double v6, 
                    double v7, double v8, double v9, double v10)
    [trivial extension for line features in 3d space]
  void free_image3_char(image3_char i);
  image3_char new_image3_char(unsigned int xsize, unsigned int ysize, unsigned int zsize);
  image3_char new_image3_char_ini( unsigned int xsize, unsigned int ysize, unsigned int zsize, unsigned char fill_value );
  image3_int new_image3_int(unsigned int xsize, unsigned int ysize, unsigned int zsize);
  image3_int new_image3_int_ini( unsigned int xsize, unsigned int ysize, unsigned int zsize, int fill_value );
  void free_image3_double(image3_double i);
  image3_double new_image3_double(unsigned int xsize, unsigned int ysize, unsigned int zsize);
  image3_double new_image3_double_ptr( unsigned int xsize, unsigned int ysize, unsigned int zsize,  double * data );
    [trivial extension for 3d data]
--------------------------------------------------------------------------------------------*/
/*
*****************************************************************************
*****************************************************************************
**Here is the header file of the original LSDSAR.
**-------------------------------------------------------------------------------------------------------
**----------------------------------------------------------------------------
LSDSAR-line segment detector for SAR images.

This code is with the publication below:

 "LSDSAR, a Markovian a contrario framework for line segment detection in SAR images",
 by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. (Pattern Recognition, 2019).

*NOTICE: This program is modified from the source code of LSD:
*"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
*Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
*Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
*http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
*Date of Modification: 27/06/2018.

*NOTICE: This program is released under GNU Affero General Public License
*and any conditions added under section 7 in the link:
*https://www.gnu.org/licenses/agpl-3.0.en.html

Copyright (c) 2017, 2018 Chenguang Liu <chenguangl@whu.edu.cn>

This program is free software: you can redistribute it and/or modify 
 it under the terms of the GNU General Public License as published 
 by the Free Software Foundation, either version 3 of the License, 
 or (at your option) any later version.

This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.
 
You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------------------

*NOTICE: This code is modified from the source code of LSD:
*"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
*Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
*Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
*http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
*Date of Modification: 27/06/2018.

The modifications lie in functions:
    1) double * lsd(int * n_out, double * img, int X, int Y),
    2) double * lsd_scale(int * n_out, double * img, int X, int Y, double scale),
    3) double * lsd_scale_region( int * n_out,
                           double * img, int X, int Y, double scale,
                           int ** reg_img, int * reg_x, int * reg_y ),
    4)double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double scale, double sigma_scale, double quant,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y ),
    5) static image_double ll_angle( image_double in, double threshold,
                              struct coorlist ** list_p, void ** mem_p,
                              image_double * modgrad, unsigned int n_bins ),
    6) static int refine( struct point * reg, int * reg_size, image_double modgrad,
                   double reg_angle, double prec, double p, struct rect * rec,
                   image_char used, image_double angles, double density_th ),
    7) static int reduce_region_radius( struct point * reg, int * reg_size,
                                 image_double modgrad, double reg_angle,
                                 double prec, double p, struct rect * rec,
                                 image_char used, image_double angles,
                                 double density_th ),
    8) static double rect_improve( struct rect * rec, image_double angles,
                            double logNT, double log_eps ),
    9) static double rect_nfa(struct rect * rec, image_double angles, double logNT),
    10) static double nfa(int n, int k, double p, double logNT).

The other functions of the code are kept unchanged.

 I would be grateful to receive any advices or possible erros in the source code. 

Chenguang Liu
Telecom ParisTech
Email: chenguang.liu@telecom-paristech.fr
Email: chenguangl@whu.edu.cn (permanent)
*/
/*
*****************************************************************************
*****************************************************************************
**Here is the header file of the original LSD.
**-------------------------------------------------------------------------------------------------------
**-------------------------------------------------------------------------------------------------------
**     
**   LSD - Line Segment Detector on digital images
**
**  This code is part of the following publication and was subject
**  to peer review:
**
**   "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
**    Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
**    Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
**    http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
**
**  Copyright (c) 2007-2011 rafael grompone von gioi <grompone@gmail.com>
**
**  This program is free software: you can redistribute it and/or modify
**  it under the terms of the GNU Affero General Public License as
**  published by the Free Software Foundation, either version 3 of the
**  License, or (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
**  GNU Affero General Public License for more details.
**
**  You should have received a copy of the GNU Affero General Public License
**  along with this program. If not, see <http://www.gnu.org/licenses/>.
**
**  ----------------------------------------------------------------------------*/


//LSD Headers
#include "lib/constants.h"
#include "lib/misc.h"
#include "lib/tuples.h"
#include "lib/markov.h"
#include "lib/interface.h"

//External headers
//#define _GNU_SOURCE
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tgmath.h>
#include <limits.h>
#include <float.h>
#include<string.h>
#include <time.h>
#include<gsl/gsl_sf_gamma.h>
#include<gsl/gsl_sf_trig.h>
#include<sys/mman.h>


/*----------------------------------------------------------------------------*/
/*------------------------------ PYTHON WRAPPER ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

/* Main Python interface.
 * Converts Python/C memory structures and decides between various LSD pipes.
 * Usage: 
 *        from pylds3 import pycis; 
 *        inputv=[alpha,eps,density,sizenum,angth,p11,p01,p11_2,p01_2,p11_4,p01_4,(scale),(sigma)]
 *        lines = pycis(I,X,Y,I0,X0,Y0,inputv,markov) 
 *
 * Input: 
 *        I - flattened X-by-Y-by-Z array of observation image data
 *              If Z=0, use 2D pipeline.  If Z>0, use 3D pipeline.
 *              
 *        I0 - flattened X0-by-Y0-by-Z0 array of noise model image data
 *              If markov==0 and X0>0, assumes I0 is a "lines" output structure prior 
 *                for centerline detection 
 *              If Z=Z0>0, will run 3D pipeline
 * 
 *        inputv - (5+6+(2))-length vector of pipeline parameters, accourding to LSDSAR
 *           alpha   - weighting parameter for Grdient-by-Ratio calulation
 *                      for extended surfaces, use alpha=4. (10-width (21x21) kernel)
 *                      for lines (thin body sufaces) use alpha=1. (3-width (7x7) kernel)
 *           eps     - NFA threshold, set to 1 by a contrario theory  
 *           density - threshold for refining bounding rectangles to a specified density, 
 *                      chosen as 0.4 default for avoiding 'nested' detections. 
 *                      for 3D surfaces with few curves, use 0.1.
 *                      Note: serves to seperate curved surfaces into linear segments.
 *                        will lead to fragmentation if too large. 
 *           sizenum - upper bound for maximum area of bounding rectangles to consider
 *                      before automatric acceptance.  Choose max[10^4,(X^2+Y^2)^(5/2)].
 *                      Note: Will be chosen internally for processing both I and I0; ignore. 
 *           angth   - absolute angular threshold for 'alignment' of points.
 *                      generally optimal at (pi/8)rads, i.e. 1/8 the full randge of orientatin
 *           px1     - Markov transition probabilities for 1, 1/2, and 1/4 angth.
 *                      May be set to 0 if I0 is present for automatic estimation 
 *           scale   - Image scaling factor for antialiasing (default 1.0, disabled)
 *                      For 2D images, use 0.8, per vonGioi.
 *           sigma   - Gaussian parameter (sigma_factor = sigma/scale) 
 *                      for antialiasing, if scale!=1.  (default 0.6 per vonGioi)
 *
 *       markov - setting for computing the conditioning pipeling (0,1,2)
 *          0 - Run LSD using prior markov statistics
 *                If X0==0, detects edges, 
 *                If X0>0, assumes I0 is prior edges and detects center lines
 *                  (only viable with 3D input Z>0)
 *          1 - Run LSD with Markov Estimation 
 *          2, 3- Run only Markov Estimation and return inputv with updated probabilities
 *                for running LSD on many images with a common noise model.
 *                2 checks for parallel alignments, 3 check for orthogonal alignments
 *
 * Output: 
 *        lines - a N-by-M list of properties for N lines, being:
 *                 If Z==0  
 *                    M=7: (x1,y1,x2,y2,width,angleth,nfa)
 *                 If Z>0
 *                    M=10: (x1,y1,z1,x2,y2,z2,widthAz,widthEl,angleth,nfa) 
 *        if markov=2 - returns inputv with updated probabilities
 *
 */
 
static PyObject * pycis(PyObject * self, PyObject * args)
{

  printf("\n\nC-MODULE OPENED\n\n");fflush(stdout);
  
  //instantiate intermediate pointers 
  PyObject * imagein; //observed image input
  PyObject * image0in; //naive image input
  PyObject * inputvin; //pipeline variables input
  PyObject * inputvorthin; //extended variables input
  double * image; //observed image
  double * image0; //naive image
  double * inputv; //pipeline variables
  double * inputvorth; //extended pipeline variables
  double * out; //output lines
  int n_points; //size variable for parsing
  int x,y,z,i,j,n; //iteration variables
  int X,Y,Z;  //image size
  int X0,Y0,Z0; //naive image size
  int markovOnly; //additional pipeline flag
  int inputv_size; //store size of inputv

  /*----------------------------------------------------------------------------------------------
  * ----------------------------------------------------------------------------------------------
  * Convert data from python to c structures 
  * ----------------------------------------------------------------------------------------------
  * ----------------------------------------------------------------------------------------------
  */

  // parse Python arguments
  if (!PyArg_ParseTuple(args, "OIIIOIIIOOI", 
    &imagein, &X, &Y, &Z, 
    &image0in, &X0, &Y0, &Z0, 
    &inputvin, &inputvorthin, &markovOnly))
    {return NULL;}
  imagein = PySequence_Fast(imagein, "arguments must be iterable");
  if(!imagein) {return 0;}
  image0in = PySequence_Fast(image0in, "arguments must be iterable");
  if(!image0in) {return 0;}
  inputvin = PySequence_Fast(inputvin, "arguments must be iterable");
  if(!inputvin) {return 0;}
  inputvorthin = PySequence_Fast(inputvorthin, "arguments must be iterable");
  if(!inputvorthin) {return 0;}
  
  // pass Python data to C structures
  n_points = PySequence_Fast_GET_SIZE(imagein);
  image = malloc(n_points*sizeof(double));
  if(!image){
    return PyErr_NoMemory( );
  }   
  for (i=0; i<n_points; i++) {
    PyObject *fitem;
    PyObject *item = PySequence_Fast_GET_ITEM(imagein, i);
    if(!item) {
      free(image);
      return 0;
    }
    fitem = PyNumber_Float(item);
    if(!fitem) {
      free(image);
      PyErr_SetString(PyExc_TypeError, "all items must be numbers");
      return 0;
    }
    image[i] = PyFloat_AS_DOUBLE(fitem);
    Py_DECREF(fitem);
  }
  n_points = PySequence_Fast_GET_SIZE(image0in);
  image0 = malloc(n_points*sizeof(double));
  if(!image0){
    return PyErr_NoMemory( );
  }   
  for (i=0; i<n_points; i++) {
    PyObject *fitem;
    PyObject *item = PySequence_Fast_GET_ITEM(image0in, i);
    if(!item) {
      free(image0);
      return 0;
    }
    fitem = PyNumber_Float(item);
    if(!fitem) {
      free(image0);
      PyErr_SetString(PyExc_TypeError, "all items must be numbers");
      return 0;
    }
    image0[i] = PyFloat_AS_DOUBLE(fitem);
    Py_DECREF(fitem);
  }
  
  n_points = PySequence_Fast_GET_SIZE(inputvin);
  inputv_size = PySequence_Fast_GET_SIZE(inputvin);
  inputv = malloc(n_points*sizeof(double));
  if(!inputv){
    return PyErr_NoMemory( );
  }   
  for (i=0; i<n_points; i++) {
    PyObject *fitem;
    PyObject *item = PySequence_Fast_GET_ITEM(inputvin, i);
    if(!item) {
      free(inputv);
      return 0;
    }
    fitem = PyNumber_Float(item);
    if(!fitem) {
      free(inputv);
      PyErr_SetString(PyExc_TypeError, "all items must be numbers");
      return 0;
    }
    inputv[i] = PyFloat_AS_DOUBLE(fitem);
    Py_DECREF(fitem);
  }   
  
  inputvorth = malloc(n_points*sizeof(double));
  if(!inputvorth){
    return PyErr_NoMemory( );
  }   
  for (i=0; i<n_points; i++) {
    PyObject *fitem;
    PyObject *item = PySequence_Fast_GET_ITEM(inputvorthin, i);
    if(!item) {
      free(inputvorth);
      return 0;
    }
    fitem = PyNumber_Float(item);
    if(!fitem) {
      free(inputvorth);
      PyErr_SetString(PyExc_TypeError, "all items must be numbers");
      return 0;
    }
    inputvorth[i] = PyFloat_AS_DOUBLE(fitem);
    Py_DECREF(fitem);
  }   


  /*----------------------------------------------------------------------------------------------
  * ----------------------------------------------------------------------------------------------
  * Run LSD
  * ----------------------------------------------------------------------------------------------
  * ----------------------------------------------------------------------------------------------
  */
  /* Pick and run the LSD pipelines
      if markovOnly==0 
        Run LSD with prior markov kernel, return lines
        if X0==0: edge line detections
        if X0>0 : center line detections (if Z>0 3D input)
      if markovOnly==1:
        Compute markov kernel and run LSD, return edge lines
      if markovOnly==2
        Compute parallel markov kernel, return updated inputv
      if markovOnly==3
        if Z>1: Compute orthogonal markov kernel, return update inputvorth
        else    Compute markovOnly==2.  Don't need orthogonal kernel
  */
  if(markovOnly==0)
  {
    /* LSD using preexisting kernel  */
    printf("Using existing Markov kernel\n");
    printf("PKernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n",
      inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);
    printf("OKernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n",
      inputvorth[5],inputvorth[6], inputvorth[7],
      inputvorth[8],inputvorth[9],inputvorth[10]);
    fflush(stdout);
    if(Z<=1) out = lsd(&n,image,X,Y,inputv,inputv_size);
    else
    {
      if(X0==0) out = lsd3b(&n,image,X,Y,Z,inputv,inputv_size,inputvorth);
      else out = lsd3centerb(&n,image,X,Y,Z,image0,X0,Y0,Z0,inputv,inputv_size,inputvorth);
    } 
  }
  else if(markovOnly==1)
  {
    /* Run full LSD+Markov pipeline  */
    printf("Estimating Markov kernel\n");
    if(Z<=1) out =  lsdM(&n,image,X,Y,image0,X0,Y0,inputv,inputv_size); 
    else     out = lsd3M(&n,image,X,Y,Z,image0,X0,Y0,Z0,inputv,inputv_size,inputvorth);
    printf("Kernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n",
      inputv[5],inputv[6], inputv[7],inputv[8],inputv[9],inputv[10]);
    fflush(stdout);

  } 
  else if(markovOnly>=2)
  {
    /* Return Markov kernel via inputv  */
    double ang_th;   /* Gradient angle tolerance in degrees.           */
    ang_th=inputv[4];
    int n_bins = 1024;       
    printf("Computing Markov kernel and returning inputv\n");
    fflush(stdout);
    if(Z0<=1) 
    {
      make_markov(image0, X0, Y0, ang_th, n_bins,inputv,inputv_size);
      out = inputv;
    }
    else
    {
      if (markovOnly==2)
      {
        make_markov3(image0, X0, Y0, Z0,  ang_th, n_bins,inputv,inputv_size,0);
        out = inputv;
      }
      else 
      {
        make_markov3(image0, X0, Y0, Z0,  ang_th, n_bins,inputvorth,inputv_size,1);
        out = inputvorth;
      }
    }
    printf("Kernel: (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f), (p11=%.4f, p10=%.4f)\n",
      out[5], out[6], out[7], out[8], out[9], out[10]);
    fflush(stdout);
  } 

  printf("\n\nCOMPLETED RUN\n\n");fflush(stdout);
  

   /*----------------------------------------------------------------------------------------------
  * ----------------------------------------------------------------------------------------------
  * Convert c to python structures
  * ----------------------------------------------------------------------------------------------
  * ----------------------------------------------------------------------------------------------
  */

  // Convert output to a valid Python structure
  // accounting for various dimensionality options
  PyObject * pyout;
  if(markovOnly>=2) //output updates variable list for markov kernel
  {
    /*Markov Kenel return */
    n_points = PySequence_Fast_GET_SIZE(inputvin);
    pyout = PyList_New((int)n_points);
    if (!pyout) {return NULL;}   
    for (i = 0; i< (int)n_points; i++) 
    {
        PyObject *num = PyFloat_FromDouble(out[i]);
        if (!num) 
        {
          Py_DECREF(pyout);
          return NULL;
        }
        PyList_SET_ITEM(pyout, i, num);
    }
  }
  else //output list of detected lines
  { 
    /*Line output return */
    int mm=7; //lines in 2D space
    if(Z>1)  mm=10; //lines in 3D space
    pyout = PyList_New((int)n*mm);
    if (!pyout) {return NULL;}   
    for (i = 0; i< n; i++) 
    {
      for(j=0;j<mm;j++) 
      {
        PyObject *num = PyFloat_FromDouble(out[i*mm+j]);
        if (!num) 
        {
	        printf("\nPYOUT ERR\n");fflush(stdout);
          Py_DECREF(pyout);
          return NULL;
        }
        PyList_SET_ITEM(pyout, i+j*n, num);
      }
    }
  }
 
  //free C memory
  free(image);
  free(image0);
  free(inputv);
  free(inputvorth); 
  //free PyObj memory
  Py_DECREF(imagein);
  Py_DECREF(image0in);
  Py_DECREF(inputvin);
  Py_DECREF(inputvorthin);
  //output
  printf("\n\nC-MODULE CLOSED\n\n");fflush(stdout);
  return pyout; 
}

//Pythonic interfaces
static PyMethodDef pycisMethods[] = {
  {"pycis", pycis, METH_VARARGS, "LSDSAR algorithm for centerline detection in 3D space"},
  {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "pycis",
  NULL,
  -1,
  pycisMethods
};

PyMODINIT_FUNC PyInit_pycis(void)
{
  return PyModule_Create(&moduledef);
}
