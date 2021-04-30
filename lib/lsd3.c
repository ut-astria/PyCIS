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
/*---------------------------------- Import ---------------------------------*/
/*----------------------------------------------------------------------------*/



#include <stdio.h>
#include <omp.h>
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

#include "lsd3.h"
#include "constants.h"
#include "misc.h"
#include "tuples.h"
#include "gradient.h"
#include "gaussians.h"
#include "markov.h"
#include "regions3.h"
#include "rectangles3.h"

/*----------------------------------------------------------------------------*/
/*------------------------ Edge Line Segment Detector ------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** 3D LSD Algotithm 
 */
double * LineSegmentDetection3( int * n_out,
                               double * img, int X, int Y, int Z,
                               double ang_th, double log_eps, double density_th,
                               int n_bins, int ** reg_img, 
                               int * reg_x, int * reg_y, int * reg_z, 
                               double * inputv, double inputv_size, double * inputvorth)
{
  printf("Launching LSD3\n");fflush(stdout);
  //instantiate variables 
  image3_double image;
  ntuple_list out = new_ntuple_list(10);
  double * return_value;
  grads angles;
  image3_double modgrad;
  image3_char used;
  struct coorlist3 * list_p;
  void * mem_p;
  struct rect3 rec;
  struct point3 * reg;
  int reg_size,min_reg_size,i;
  unsigned int xsize,ysize,zsize;
  angles3 reg_angle = new_angles3(0.,0.);
  double prec,p,log_nfa,logNT;
  double scale,sigma_scale;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */
  int ls_total = 0;
  double beta=inputv[0];
  int sizenum=(int) inputv[3];

  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 || Z<=0 ) error("invalid image3 input.");
  if( ang_th <= 0.0 || ang_th >= 180.0 )
    error("'ang_th' value must be in the range (0,180).");
  if( density_th < 0.0 || density_th > 1.0 )
    error("'density_th' value must be in the range [0,1].");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");

  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;

  /* load and scale image (if necessary) and compute angle at each pixel */
  //set scale defaults 
  if (inputv_size>11)  scale=(double)inputv[11];
  else scale=1.0;
  if (inputv_size>12)  sigma_scale=(double)inputv[12];
  else sigma_scale=0.6;
  //optional gaussian scaling 
  image = new_image3_double_ptr((unsigned int)X, (unsigned int)Y, (unsigned int)Z, img);
  if (scale != 1.0) 
  {
    sigma_scale = (double) inputv[12];
    image3_double scaled_image;
    scaled_image = gaussian3_sampler(image,scale,sigma_scale);
    angles = ll_angle3( scaled_image, &list_p, &mem_p, &modgrad,
		       (unsigned int) n_bins,beta);
    free_image3_double(scaled_image);
  }
  else angles = ll_angle3( image, &list_p, &mem_p, &modgrad,
		       (unsigned int) n_bins,beta);
  printf("LLangle finished successfully\n");fflush(stdout);
  //set size variables 
  xsize = angles->az->xsize;
  ysize = angles->az->ysize;
  zsize = angles->az->zsize;

  /* Number of Tests - NT

    The theoretical number of tests is Np.(XY)^(5/2) in two-dimensional space.
    where X and Y are number of columns and rows of the image.
    Np corresponds to the number of angle precisions considered.
    
    We consider Np=3 per Liu et al.
    In the 3D case, there are XYZ options for each endpoint.
    Furthermore, (XYZ)^{1/3} width options are considered in the two tangent bases
    Then we have Np*XYZ*XYZ*XYZ^{1/3}*XYZ^{1/3} =Np*(XYZ)^{8/3}
    Therefore, the runtime cost is higher by O[(XY)^{16/15}*Z^{8/3}]
    NT(3D)/NT(2D)=~O(Z^{8/3}) For a uniform cube 
    
  */
  printf("Creating regions, used, and mnfa matrices\n");fflush(stdout);
  logNT = 8.0 * (log10((double)xsize) + log10((double)ysize) + log10((double)zsize)) / 3.0
          + log10(3.0);
  
  /* initialize some structures */
  used = new_image3_char_ini(xsize,ysize,zsize,NOTUSED);
  reg = (struct point3 *) calloc( (size_t) (xsize*ysize*zsize), sizeof(struct point3) );
  if( reg == NULL ) error("not enough memory!");

  // instantiate markov variables
  double p0,p1,p11,p10,p01,p00;
  int N=sizenum;
  int NOUT=N;
  int min_reg_size_2, min_reg_size_4;
  double * output = ( double *) malloc(N*N*sizeof(double));
  double * output_2 = ( double *) malloc(N*N*sizeof(double));
  double * output_4 = ( double *) malloc(N*N*sizeof(double));
  double * outputP = (double *)malloc(3*sizeof(double));
  double * outputP_2 = (double *)malloc(3*sizeof(double));
  double * outputP_4 = (double *)malloc(3*sizeof(double));

  //solve markov transition matrices using kernels
  printf("Creating markov tranition matrices \n");fflush(stdout);
  p11=inputv[5];  p01=inputv[6];
  p1=p;  p0=1-p1;
  outputP[0]=p0; outputP[1]=p11; outputP[2]=p01;
  min_reg_size=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
  NFA_matrix(output,p0,p11,p01,N);
  p11=inputv[7];  p01=inputv[8];
  p1=p/2;  p0=1-p1;
  outputP_2[0]=p0; outputP_2[1]=p11; outputP_2[2]=p01;
  min_reg_size_2=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1; 
  NFA_matrix(output_2,p0,p11,p01,N);
  p11=inputv[9];  p01=inputv[10];
  p1=p/4;  p0=1-p1;
  outputP_4[0]=p0; outputP_4[1]=p11; outputP_4[2]=p01;
  min_reg_size_4=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
  NFA_matrix(output_4,p0,p11,p01,N);
  
  //Instantiate angle images 
  printf("Instantiating angle images\n");fflush(stdout);
  double * azimg = (double *) malloc(xsize*ysize*zsize*sizeof(double));
  double * elimg = (double *) malloc(xsize*ysize*zsize*sizeof(double));
  for(i=0;i<(xsize*ysize*zsize);i++)
  {
    azimg[i]= angles->az->data[i];
    elimg[i]= angles->el->data[i];
  }
  
  //Instantiate time variables 
  double startT,endT,growtime,regiontime,refinetime,improvetime;
  int NOUT2=(int) xsize*ysize*zsize / 1;
  double timeall = omp_get_wtime();
  /*Timeout check in seconds, will test each loop*/
  int printout=0;
  double timelim = 60.*60.;//10.*60.;



   /*
   -------------------------------------------------------------------------------------------------------------
   Begin line segment search 
   -------------------------------------------------------------------------------------------------------------
   */



  /* search for line segments */
  printf("Searching for line segments...\n"); fflush(stdout);
  for(; list_p != NULL; list_p = list_p->next )
  { 
    if( ((omp_get_wtime()-timeall)<timelim) && 
        used->data[ list_p->z + (list_p->x + list_p->y *xsize) * zsize ] == NOTUSED &&
        azimg[ list_p->z + (list_p->x + list_p->y * xsize)*zsize ] != NOTDEF  &&
        elimg[ list_p->z + (list_p->x + list_p->y * xsize) *zsize ] != NOTDEF  )
       /* there is no risk of double comparison problems here
          because we are only interested in the exact NOTDEF value */
    {
      fflush(stdout);
      /* find the region of connected point and ~equal angle */
      region3_grow( list_p->x, list_p->y, list_p->z, angles, reg, &reg_size,
                     &reg_angle, used, prec, NOUT2);
      
      /* reject small regions */
      if( reg_size < min_reg_size ){continue;}

      /* construct rectangular approximation for the region */
      region2rect3(reg,reg_size,modgrad,reg_angle,prec,p,&rec,0);
    
      /* Check if the rectangle exceeds the minimal density of
         region points. If not, try to improve the region.
         The rectangle will be rejected if the final one does
         not fulfill the minimal density condition.
         This is an addition to the original LSD algorithm published in
         "LSD: A Fast Line Segment Detector with a False Detection Control"
         by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
         The original algorithm is obtained with density_th = 0.0.
      */
      if( !refine3( reg, &reg_size, modgrad, reg_angle,
                    prec, p, &rec, used, angles,density_th,NOUT2,0 ) ) {continue;}
    
      /* compute NFA value */ 
      log_nfa = rect3_improve(&rec,angles,logNT,log_eps,output,output_2,output_4,
                              outputP,outputP_2,outputP_4, 
                              NOUT,min_reg_size,min_reg_size_2,min_reg_size_4,0);
    
      if( log_nfa <= log_eps ) {continue;}

      /* A New Line Segment was found! */
      ++ls_count;  /* increase line segment counter */
      

      /*Correct for gaussian scaling */
      if(scale != 1.0)
      {
        rec.x1/=scale; rec.y1/=scale; rec.z1/=scale;
        rec.x2/=scale; rec.y2/=scale; rec.z2/=scale;
        rec.width1/=scale; rec.width2/=scale;
        rec.length = dist3(rec.x1,rec.y1,rec.z1,rec.x2,rec.y2,rec.z2);
      }
      /* add line segment found to output */
      //printf("grow: %.4f, region: %.4f, refine: %.4f, improve: %.4f\n",growtime,regiontime,refinetime,improvetime);fflush(stdout);
      //printf("\t LINE: NFA %.2f, lww: (%.2f, %.2f, %.2f), azel: (%.2f, %.2f), xyz1 (%d,%d,%d)...\n",log_nfa,rec.length,rec.width1,rec.width2, rec.theta->az*180./M_PI, rec.theta->el*180./M_PI,(int)rec.x1,(int)rec.y1,(int)rec.z1);  fflush(stdout);
      add_10tuple( out, rec.x1, rec.y1, rec.z1, rec.x2, rec.y2, rec.z2, 
                         rec.width1, rec.width2, rec.p, log_nfa );
    } 
    else if( ((omp_get_wtime()-timeall)>=timelim) && (printout==0))
    {
      printf("\nTIMEOUT %.2f\n",omp_get_wtime()-timeall);fflush(stdout);printout=1;
    } 
  }
  printf("\nCONCLUDED FULL LOOP------------------------\n");fflush(stdout);
  printf("\nSTAGE1: %d lines in %.2f\n",ls_count,omp_get_wtime()-timeall);fflush(stdout); 
  
  /*
  -------------------------------------------------------------------------------------------------------------
  Free memory and return result
  -------------------------------------------------------------------------------------------------------------
  */



  /* only the double_image structure should be freed,
    the data pointer was provided to this functions
    and should not be destroyed.                 */
  free( (void *) image );   
  free_grads(angles);
  free_angles3(reg_angle);
  free_image3_double(modgrad);
  free_image3_char(used);
  free( (void *) reg );
  free( (void *) mem_p );
  free((void *) output); 
  free((void *) output_2);
  free((void *) output_4);
  free((void *) outputP); 
  free((void *) outputP_2);
  free((void *) outputP_4);
  free(azimg);
  free(elimg);

  /*switch edge or centerline output*/
  /* only the 'ntuple_list' structure must be freed,
    but the 'values' pointer must be keep to return
    as a result. */
  if( out->size > (unsigned int) INT_MAX )
    error("too many detections to fit in an INT.");
  *n_out = (int) (out->size);
  return_value = out->values;
  free( (void *) out ); 
  return return_value;
}



/*----------------------------------------------------------------------------*/
/*----------------------- Center Line Segment Detector -----------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** 3D LSD Algotithm 
 */
double * LineSegmentDetection3Center( int * n_out,
                               double * img, int X, int Y, int Z,
                               double * img0, int X0, int Y0, int Z0,
                               double ang_th, double log_eps, double density_th,
                               int n_bins, int ** reg_img, 
                               int * reg_x, int * reg_y, int * reg_z, 
                               double * inputv, double inputv_size, double * inputvorth)
{
  printf("Launching LSD3\n");fflush(stdout);
  //instantiate variables 
  image3_double image;
  ntuple_list out = new_ntuple_list(10);
  ntuple_list out2 = new_ntuple_list(10);
  double * return_value;
  grads angles;
  image3_double modgrad;
  image3_char used;
  struct coorlist3 * list_p;
  void * mem_p;
  struct rect3 rec;
  struct point3 * reg;
  int reg_size,min_reg_size,i;
  unsigned int xsize,ysize,zsize;
  angles3 reg_angle = new_angles3(0.,0.);
  double prec,p,log_nfa,logNT;
  double scale,sigma_scale;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */
  int ls_total = 0;
  double beta=inputvorth[0];
  int sizenum=(int) inputvorth[3];

  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 || Z<=0 ) error("invalid image3 input.");
  if( ang_th <= 0.0 || ang_th >= 180.0 )
    error("'ang_th' value must be in the range (0,180).");
  if( density_th < 0.0 || density_th > 1.0 )
    error("'density_th' value must be in the range [0,1].");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");

  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;

  /* load and scale image (if necessary) and compute angle at each pixel */
  //set scale defaults 
  if (inputv_size>11)  scale=(double)inputvorth[11];
  else scale=1.0;
  if (inputv_size>12)  sigma_scale=(double)inputvorth[12];
  else sigma_scale=0.6;
  //optional gaussian scaling 
  image = new_image3_double_ptr((unsigned int)X, (unsigned int)Y, (unsigned int)Z, img);
  if (scale != 1.0) 
  {
    sigma_scale = (double) inputvorth[12];
    image3_double scaled_image;
    scaled_image = gaussian3_sampler(image,scale,sigma_scale);
    angles = ll_angle3( scaled_image, &list_p, &mem_p, &modgrad,
		       (unsigned int) n_bins,beta);
    free_image3_double(scaled_image);
  }
  else angles = ll_angle3( image, &list_p, &mem_p, &modgrad,
		       (unsigned int) n_bins,beta);
  printf("LLangle finished successfully\n");fflush(stdout);
  //set size variables 
  xsize = angles->az->xsize;
  ysize = angles->az->ysize;
  zsize = angles->az->zsize;

  /* Number of Tests - NT

    The theoretical number of tests is Np.(XY)^(5/2) in two-dimensional space.
    where X and Y are number of columns and rows of the image.
    Np corresponds to the number of angle precisions considered.
    
    We consider Np=3 per Liu et al.
    In the 3D case, there are XYZ options for each endpoint.
    Furthermore, (XYZ)^{1/3} width options are considered in the two tangent bases
    Then we have Np*XYZ*XYZ*XYZ^{1/3}*XYZ^{1/3} =Np*(XYZ)^{8/3}
    Therefore, the runtime cost is higher by O[(XY)^{16/15}*Z^{8/3}]
    NT(3D)/NT(2D)=~O(Z^{8/3}) For a uniform cube 
    
  */
  printf("Creating regions, used, and mnfa matrices\n");fflush(stdout);
  logNT = 8.0 * (log10((double)xsize) + log10((double)ysize) + log10((double)zsize)) / 3.0
          + log10(3.0);
  
  /* initialize some structures */
  //effectively resets exclusion principle for centerline detection 
  used = new_image3_char_ini(xsize,ysize,zsize,NOTUSED);
  reg = (struct point3 *) calloc( (size_t) (xsize*ysize*zsize), sizeof(struct point3) );
  if( reg == NULL ) error("not enough memory!");

  // instantiate markov variables
  double p0,p1,p11,p10,p01,p00;
  int N=sizenum;
  int NOUT=N;
  int min_reg_size_2, min_reg_size_4;
  double * output = ( double *) malloc(N*N*sizeof(double));
  double * output_2 = ( double *) malloc(N*N*sizeof(double));
  double * output_4 = ( double *) malloc(N*N*sizeof(double));
  double * outputP = (double *)malloc(3*sizeof(double));
  double * outputP_2 = (double *)malloc(3*sizeof(double));
  double * outputP_4 = (double *)malloc(3*sizeof(double));
  
  printf("Creating markov tranition matrices \n");fflush(stdout);
  p11=inputvorth[5];  p01=inputvorth[6];
  p1=p;  p0=1-p1;
  outputP[0]=p0; outputP[1]=p11; outputP[2]=p01;
  min_reg_size=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
  NFA_matrix(output,p0,p11,p01,N);
  p11=inputvorth[7];  p01=inputvorth[8];
  p1=p/2;  p0=1-p1;
  outputP_2[0]=p0; outputP_2[1]=p11; outputP_2[2]=p01;
  min_reg_size_2=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1; 
  NFA_matrix(output_2,p0,p11,p01,N);
  p11=inputvorth[9];  p01=inputvorth[10];
  p1=p/4;  p0=1-p1;
  outputP_4[0]=p0; outputP_4[1]=p11; outputP_4[2]=p01;
  min_reg_size_4=(int) (-log_eps-logNT-log10(p1))/log10(p11)+1;
  NFA_matrix(output_4,p0,p11,p01,N);
  printf("success!\n");fflush(stdout);

  
  //Instantiate angle images 
  printf("Instantiating angle images\n");fflush(stdout);
  double * azimg = (double *) malloc(xsize*ysize*zsize*sizeof(double));
  double * elimg = (double *) malloc(xsize*ysize*zsize*sizeof(double));
  for(i=0;i<(xsize*ysize*zsize);i++)
  {
    azimg[i]= angles->az->data[i];
    elimg[i]= angles->el->data[i];
  }
  
  //Instantiate time variables 
  double startT,endT,growtime,regiontime,refinetime,improvetime;
  int NOUT2=(int) xsize*ysize*zsize / 1;
  double timeall = omp_get_wtime();
  /*Timeout check in seconds, will test each loop*/
  int printout=0;
  double timelim = 60.*60.;//10.*60.;



  /*
  -------------------------------------------------------------------------------------------------------------
  Import prior edge lines from double* to tuple
  -------------------------------------------------------------------------------------------------------------
  */



  ls_count = X0/10;
  for(i=0; i<ls_count; i++)
  {

    add_10tuple( out, img0[i+0*ls_count], img0[i+1*ls_count], img0[i+2*ls_count],
      img0[i+3*ls_count], img0[i+4*ls_count], img0[i+5*ls_count], img0[i+6*ls_count],
      img0[i+7*ls_count], img0[i+8*ls_count], img0[i+9*ls_count] );
  }



   /*
   -------------------------------------------------------------------------------------------------------------
   Begin line segment search, centerline detection 
   -------------------------------------------------------------------------------------------------------------
   */
  
 
  //Redefine markov transition matrices for orthogonal-alignment kernel
  
  //INSTANTIATION
  int lx,ly,lz,lx2,ly2,lz2;
  int tempx, tempy, tempz;
  int ls_count2=0; 
  int ls_count2reject=0;
  density_th = 0.;
  angles3 lstheta;

  //Iterate over edge lines - no centerlines can meaningfully exist without edges
  for(int lidx=0; lidx<ls_count; lidx++)
  { 

    //instantiate core as first endpoint
    lx = out->values[ lidx * out->dim + 0];
    ly = out->values[ lidx * out->dim + 1];
    lz = out->values[ lidx * out->dim + 2];
    lx2 = out->values[ lidx * out->dim + 3];
    ly2 = out->values[ lidx * out->dim + 4];
    lz2 = out->values[ lidx * out->dim + 5];
    //check endpoint order and avoid edge anomalies
    // if both endpoints outside image domain, skip entierly (edge events)
    // else, start from the endpoint inside the image domain 
    if ( (lx<=0.)||(ly<=0.)||(lz<=0.)||
    (lx>=xsize)||(ly>=ysize)||(lz>=zsize))
    {
      if ( (lx2<=0.)||(ly2<=0.)||(lz2<=0.)||
        (lx2>=xsize)||(ly2>=ysize)||(lz2>=zsize))
      {continue;} //if both points outside valid domain, skip
      tempx = lx; tempy = ly; tempz = lz;
      lx = lx2; ly = ly2; lz = lz2;
      lx2 = tempx; lz2 = tempz; lz2 = tempz;
    }

    //Begin main iteration
    if( ((omp_get_wtime()-timeall)<timelim) && 
        used->data[ lz + (lx + ly  * xsize) * zsize ] == NOTUSED &&
        azimg[ lz + (lx + ly * xsize) * zsize ] != NOTDEF  &&
        elimg[ lz + (lx + ly * xsize) * zsize ] != NOTDEF  )
    {

      //get expected centerline from prior
      lstheta = line_angle3(lx,ly,lz,lx2,ly2,lz2);  

      /* find the region of connected point and ~equal angle */
      region3_growORTH( lx, ly, lz, modgrad, angles, reg, &reg_size,
                    &reg_angle, &lstheta, used, prec, NOUT2);

      /* reject small regions */
      if( reg_size < min_reg_size )
      {
        ++ls_count2reject;
        continue;
      }
      
      /* construct rectangular approximation for the region */
      region2rect3(reg,reg_size,modgrad,reg_angle,prec,p,&rec,1);
      
      /* Check if the rectangle exceeds the minimal density */
      if( !refine3( reg, &reg_size, modgrad, reg_angle,
                    prec, p, &rec, used, angles,density_th,NOUT2,1 ) ) 
      { 
        ++ls_count2reject;
        continue;
      }

      /* compute NFA value */ 
      log_nfa = rect3_improve(&rec,angles,logNT,log_eps,output,output_2,output_4,
                              outputP,outputP_2,outputP_4,
                              NOUT,min_reg_size,min_reg_size_2,min_reg_size_4,1);

      rec.length = dist3(rec.x1,rec.y1,rec.z1,rec.x2,rec.y2,rec.z2);
  
      if( log_nfa <= log_eps ) 
      {
        ++ls_count2reject;
        continue;
      }

      /* A New Line Segment was found! */
      ++ls_count2;  /* increase line segment counter */
      
      /* add line segment found to output */
      if(scale != 1.0)
      {
        rec.x1/=scale; rec.y1/=scale; rec.z1/=scale;
        rec.x2/=scale; rec.y2/=scale; rec.z2/=scale;
        rec.width1/=scale; rec.width2/=scale;
      }
      add_10tuple( out2, rec.x1, rec.y1, rec.z1, rec.x2, rec.y2, rec.z2, 
                        rec.width1, rec.width2, rec.p, log_nfa );
    }

    else if( ((omp_get_wtime()-timeall)>=timelim) && (printout==0))
    {
      printf("\nTIMEOUT %.2f\n",omp_get_wtime()-timeall);fflush(stdout);printout=1;
    } 
  }

  printf("\nSTAGE2 %d/%d updated, %d/%d originals, %d/%d deleted... in %.2f\n",ls_count2,ls_count,ls_count2reject,ls_count,(ls_count-ls_count2-ls_count2reject),ls_count,omp_get_wtime()-timeall);fflush(stdout); 



  /*
  -------------------------------------------------------------------------------------------------------------
  Free memory and return result
  -------------------------------------------------------------------------------------------------------------
  */



  /* only the double_image structure should be freed,
    the data pointer was provided to this functions
    and should not be destroyed.                 */
  free_angles3(lstheta);
  free( (void *) image );   
  free_grads(angles);
  free_angles3(reg_angle);
  free_image3_double(modgrad);
  free_image3_char(used);
  free( (void *) reg );
  free( (void *) mem_p );
  free((void *) output); 
  free((void *) output_2);
  free((void *) output_4);
  free((void *) outputP); 
  free((void *) outputP_2);
  free((void *) outputP_4);
  free(azimg);
  free(elimg);

  /*switch edge or centerline output*/
  /* only the 'ntuple_list' structure must be freed,
    but the 'values' pointer must be keep to return
    as a result. */
 
  if( out2->size > (unsigned int) INT_MAX )
    error("too many detections to fit in an INT.");
  *n_out = (int) (out2->size);
  return_value = out2->values;
  
  free( (void *) out );
  free( (void *) out2 );  
  return return_value;
}
