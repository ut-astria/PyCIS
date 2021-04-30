/*----------------------------------------------------------------------------*/
/*---------------------------------- Import ---------------------------------*/
/*----------------------------------------------------------------------------*/

#ifndef GSOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tgmath.h>
#include <limits.h>
#include <float.h>
#include<string.h>
#include <time.h>
#include<gsl/gsl_eigen.h>
#include<gsl/gsl_sf_gamma.h>
#include<gsl/gsl_sf_trig.h>
#include<sys/mman.h>

#include "regions3.h"
#include "constants.h"
#include "misc.h"
#include "tuples.h"
#include "nfa.h"
#include "rectangles3.h"


/*----------------------------------------------------------------------------*/
/*---------------------------------- Regions ---------------------------------*/
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.
    The following is the region inertia matrix A:
    @f[
        A = \left(\begin{array}{cc}
                Ixx & Ixy & Ixz\\
                Ixx & Iyy & Iyz\\
                Ixz & Iyz & Izz
             \end{array}\right)
    @f]
    where
      Ixx =   sum_i G(i).((y_i - cy)^2 + (z_i - cz)^2)
      Iyy =   sum_i G(i).((x_i - cx)^2 + (z_i - cz)^2)
      Izz =   sum_i G(i).((x_i - cx)^2 + (y_i - cy)^2)
      Ixy = - sum_i G(i).(x_i - cx).(y_i - cy)
      Ixz = - sum_i G(i).(x_i - cx).(z_i - cz)
      Ixy = - sum_i G(i).(y_i - cy).(z_i - cz)
    and
    - G(i) is the gradient norm at pixel i, used as pixel's weight.
    - x_i, y_i, z_i are the coordinates of pixel i.
    - cx, cy, cz are the coordinates of the center of th region.

    Eigendecomposition by GSL for real symmetric matrices.
    Returns az/el of the principal eigenvector for region, 
    which will be ORTHOGONAL to aligned pixelsn gradient vectors.  
    Intermediate and minor eigenvector orientations 
    are returned as az2/el2 and az3/el3 respectively.
*/

angles3 get_theta3( struct point3 * reg, int reg_size, double x, double y, double z,
                         image3_double modgrad, angles3 reg_angle, double prec, int orth )
{
  fflush(stdout);
  double lambda,weight;
  double Ixx = 0.0;
  double Iyy = 0.0;
  double Izz = 0.0;
  double Ixy = 0.0;
  double Ixz = 0.0;
  double Iyz = 0.0;
  int i;


  /* check parameters */
  if( reg == NULL ) error("get_theta3: invalid region.");
  if( reg_size <= 1 ) error("get_theta3: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("get_theta3: invalid 'modgrad'.");
  if( prec < 0.0 ) error("get_theta3: 'prec' must be positive.");
  
  /* compute inertia matrix */
  for(i=0; i<reg_size; i++)
    {
    
      weight = modgrad->data[reg[i].z + (reg[i].x + reg[i].y * modgrad->xsize)
          * modgrad->zsize ];
      Ixx += ( ( (double) reg[i].y - y ) * ( (double) reg[i].y - y ) + 
           ( (double) reg[i].z - z ) * ( (double) reg[i].z - z) )
          * weight;
      Iyy += ( ( (double) reg[i].x - x ) * ( (double) reg[i].x - x ) + 
           ( (double) reg[i].z - z ) * ( (double) reg[i].z - z) )
          * weight;
      Izz += ( ( (double) reg[i].y - y ) * ( (double) reg[i].y - y ) + 
           ( (double) reg[i].x - x ) * ( (double) reg[i].x - x) )
          * weight;  
      Ixy -= ( (double) reg[i].x - x ) * ( (double) reg[i].y - y ) * weight;
      Ixz -= ( (double) reg[i].x - x ) * ( (double) reg[i].z - z ) * weight;
      Iyz -= ( (double) reg[i].z - z ) * ( (double) reg[i].y - y ) * weight;
  
    }
  if( double_equal(Ixx,0.0) && double_equal(Iyy,0.0) && double_equal(Ixy,0.0) &&
    double_equal(Izz,0.0) && double_equal(Ixz,0.0) && double_equal(Iyz,0.0))
    error("get_theta3: null inertia matrix.");

  /*Gsl eigendecomposition on real symmetric data, for robustness 
   *'symmetric bidiagonalization and QR reduction method' per documentation 
   *eigenvalues accurate up to e_mach*||I||_2 
   */  
  size_t dim = 3;
  gsl_eigen_symmv_workspace * ework = gsl_eigen_symmv_alloc(dim);
  gsl_matrix * Imat = gsl_matrix_alloc(dim,dim);
  gsl_matrix_set(Imat, 0, 0, Ixx);
  gsl_matrix_set(Imat, 1, 1, Iyy);
  gsl_matrix_set(Imat, 2, 2, Izz);
  gsl_matrix_set(Imat, 0, 1, Ixy);
  gsl_matrix_set(Imat, 1, 0, Ixy);
  gsl_matrix_set(Imat, 0, 2, Ixz);
  gsl_matrix_set(Imat, 2, 0, Ixz);
  gsl_matrix_set(Imat, 1, 2, Iyz);
  gsl_matrix_set(Imat, 2, 1, Iyz);
  gsl_vector * eval = gsl_vector_alloc(dim);
  gsl_matrix * evec = gsl_matrix_alloc(dim,dim);
  
  //Get principal eigenvector
  int out = gsl_eigen_symmv(Imat, eval, evec, ework);
  //gsl_eigen_symmv-sort(eigenvalues, eigenvectors, GSL_EIGEN_SORT_ABS_DESV)
  if(~gsl_vector_ispos(eval)) {for(i=0;i<3;i++) gsl_vector_set(eval,i,fabs(gsl_vector_get(eval,i)));}
  size_t idx = gsl_vector_min_index(eval);
  double xv = gsl_matrix_get(evec, 0, idx);
  double yv = gsl_matrix_get(evec, 1, idx);
  double zv = gsl_matrix_get(evec, 2, idx);
  angles3 theta = line_angle3(0,0,0,xv,yv,zv);  
  //Get minor eigenvector
  size_t idx3 = gsl_vector_max_index(eval);
  xv = gsl_matrix_get(evec, 0, idx3);
  yv = gsl_matrix_get(evec, 1, idx3);
  zv = gsl_matrix_get(evec, 2, idx3);
  angles3 theta3 = line_angle3(0,0,0,xv,yv,zv);
  theta->az3 = (double) theta3->az; theta->el3 = (double) theta3->el;
  //Get intermediate eigenvector (remaining index)
  size_t idx2;
  if(((idx==0) && (idx3==1)) || ((idx==1) && (idx3==0)))  idx2=2;
  else if (((idx==0) && (idx3==2)) || ((idx==2) && (idx3==0))  ) idx2=1;
  else if (((idx==1) && (idx3==2)) ||  ((idx==2) && (idx3==1)) )  idx2=0;    
  xv = gsl_matrix_get(evec, 0, idx2);
  yv = gsl_matrix_get(evec, 1, idx2);
  zv = gsl_matrix_get(evec, 2, idx2);
  angles3 theta2 = line_angle3(0,0,0,xv,yv,zv);  
  theta->az2 = (double) theta2->az; theta->el2 = (double) theta2->el;
  //free temp data
  free_angles3(theta2);
  free_angles3(theta3);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
  gsl_matrix_free(Imat);
  gsl_eigen_symmv_free(ework);


  //For orthogonal case (centerline anlysis), confirm az/el principal is aligned to prior
  if(orth==1)
  {
    double tempaz, tempel;
    if( !isaligned3(theta->az,theta->el,reg_angle->az,reg_angle->el,prec)) 
    {
      //printf("\t\tGET_THETA3: principal unaligned with prior.  Attempting correction...\n"); fflush(stdout);
      if( isaligned3(theta->az2,theta->el2,reg_angle->az,reg_angle->el,prec))
      {
        tempaz = theta->az; tempel = theta->el;
        theta->az = theta->az2; theta->el = theta->el2;
        theta->az2 = tempaz; theta->el2 = tempel;
      } 
      else if( isaligned3(theta->az3,theta->el3,reg_angle->az,reg_angle->el,prec)) 
      {
        tempaz = theta->az; tempel = theta->el;
        theta->az = theta->az3; theta->el = theta->el3;
        theta->az3 = tempaz; theta->el3 = tempel;
      }
    }
  }

  return theta;
}

/*----------------------------------------------------------------------------*/
/** Computes a 3D rectangular prism that covers a region of points.
 */
void region2rect3( struct point3 * reg, int reg_size,
                         image3_double modgrad, angles3 reg_angle,
                         double prec, double p, struct rect3 * rec , int orth )
{
  double x,y,z,dx,dy,dz;
  double l,w1,w2,weight,sum,l_min,l_max,w1_min,w1_max,w2_min,w2_max;
  angles3 theta;
  int i;

  /* check parameters */
  if( reg == NULL ) error("region2rect3: invalid region.");
  if( reg_size <= 1 ) error("region2rect3: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("region2rect3: invalid image 'modgrad'.");
  if( rec == NULL ) error("region2rect3: invalid 'rec'.");
  if((orth!=0)&&(orth!=1))error("recgion2rect3: invalid orth flag");
  
  /* weighed center of the region */
  x = 0; y = 0; z = 0; sum = 0.0;
  for(i=0; i<reg_size; i++)
  {
    weight = modgrad->data[ reg[i].z + (reg[i].x + reg[i].y * modgrad->xsize) 
        * modgrad->zsize ];
    x += (double) reg[i].x * weight;
    y += (double) reg[i].y * weight;
    z += (double) reg[i].z * weight;
    sum += weight;
  }
  if( sum <= 0.0 ) error("region2rect3: weights sum equal to zero.");
  x /= sum;
  y /= sum;
  z /= sum;

  /* theta */
  theta = get_theta3(reg,reg_size,x,y,z,modgrad,reg_angle,prec,orth);

  /* length and width:

     (dx,dy,dx) are computed by standard polar-cartesian relations.
     'l' is defined along the primary principal axis, 
     and 'w1','w2' along the azimuth/elevation tangents of the principal axis.
     Accordingly, where c and s abbreviate sine,cosine:
     lhat  = dR/dr  = [caz*sel, saz*sel,  cel]
     w1hat = dR/daz = [   -saz,     cel,    0]
     w2hat = dR/del = [caz*cel, saz*cel, -sel] 

     The length of the rectangle goes from l_min to l_max, where l_min
     and l_max are the minimum and maximum values of l in the region.
     Analogously, the width is selected from w_min to w_max, where
     w_min and w_max are the minimum and maximum of w for the pixels
     in the region.
   */
  double saz,caz,sel,cel;
  sincos(theta->az,&saz,&caz);
  sincos(theta->el,&sel,&cel);
  double  dl[3] =  {caz*sel, saz*sel,  cel};
  double dw1[3] =  {-saz, caz, 0};
  double dw2[3] =  {caz*cel, saz*cel, -sel};

  l_min = 0.;l_max = 0.;w1_min = 0.;w1_max =0.; w2_min =0.; w2_max = 0.0;
  for(i=0; i<reg_size; i++)
  {
    l =  ( (double) reg[i].x - x) * dl[0] + ( (double) reg[i].y - y) * dl[1] + 
       ( (double) reg[i].z - z) * dl[2];
    w1=  ( (double) reg[i].x - x) * dw1[0] + ( (double) reg[i].y - y) * dw1[1] + 
       ( (double) reg[i].z - z) * dw1[2];
    w2=  ( (double) reg[i].x - x) * dw2[0] + ( (double) reg[i].y - y) * dw2[1] + 
       ( (double) reg[i].z - z) * dw2[2];

    if( l > l_max ) l_max = l;
    if( l < l_min ) l_min = l;
    if( w1 > w1_max ) w1_max = w1;
    if( w1 < w1_min ) w1_min = w1;
    if( w2 > w2_max ) w2_max = w2;
    if( w2 < w2_min ) w2_min = w2;
  }

  /* store values */
  rec->x1 = x + l_min * dl[0];
  rec->y1 = y + l_min * dl[1];
  rec->z1 = z + l_min * dl[2];
  rec->x2 = x + l_max * dl[0];
  rec->y2 = y + l_max * dl[1];
  rec->z2 = z + l_max * dl[2];
  rec->length = l_max - l_min;
  rec->width1 = w1_max - w1_min;
  rec->width2 = w2_max - w2_min;
  rec->x = x;
  rec->y = y;
  rec->z = z;
  rec->theta = theta;
  for(i=0;i<3;i++)
  {
    rec->dl[i]  = (double)dl[i];
    rec->dw1[i] = (double)dw1[i];
    rec->dw2[i] = (double)dw2[i];
  }
  rec->prec = prec;
  rec->p = p;

  /* we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
   */
  if( rec->width1 < 1.0 ) rec->width1 = 1.0;
  if( rec->width2 < 1.0 ) rec->width2 = 1.0;
}



/*----------------------------------------------------------------------------*/
/** Build a 3D region of pixels that share the same angle, up to a
    tolerance 'prec', starting at point (x,y,z). 
    
    Region growing is constrained to parallel 3D orientations, 
    since we would require the principal axis a-prior 
    to grow based on orthogonal orientation to said axis.   
*/
void region3_grow(int x, int y,int z, grads angles, 
                         struct point3 * reg,
                         int * reg_size, angles3 * reg_angle, 
                         image3_char used,double prec ,int NOUT)
{
    
  double sumdx,sumdy,sumdz;
  int xx,yy,zz,i;

  /* check parameters */
  if( x < 0 || y < 0 || z<0 || x >= (int) used->xsize 
          || y >= (int) used->ysize || z >= (int) used->zsize)
    error("region3_grow: (x,y,z) out of the image.");
  if( angles->az == NULL || angles->az->data == NULL )
    error("region3_grow: invalid grads 'angles' or image 'az'.");
  if( reg == NULL ) error("region3_grow: invalid 'reg'.");
  if( reg_size == NULL ) error("region3_grow: invalid pointer 'reg_size'.");
  if( reg_angle == NULL ) error("region3_grow: invalid pointer 'reg_angle'.");
  if( used == NULL || used->data == NULL )
    error("region3_grow: invalid image 'used'.");

  /* first point of the region */
  *reg_size = 1;
  reg[0].x = x;
  reg[0].y = y;
  reg[0].z = z;
  /*regions angles*/
  int xsize = (int)used->xsize;
  int ysize = (int)used->ysize;
  int zsize = (int)used->zsize;
  //get angle
  double reg_az, reg_el;
  reg_az = angles->az->data[z + (x+y*xsize)*zsize]; 
  reg_el = angles->el->data[z + (x+y*xsize)*zsize]; 
  //compute gradient direction in cartesian coordinates
  double saz,caz,sel,cel;       
  sincos(reg_az,&saz,&caz);
  sincos(reg_el,&sel,&cel);
  sumdx = caz*sel;
  sumdy = saz*sel;
  sumdz = cel;
  used->data[z+(x+y*xsize)*zsize] = USED;

  /* try neighbors as new region points */
  double grads_az,grads_el;
  double prectemp = prec;
  for(i=0; i<*reg_size; i++)
    for(zz=reg[i].z-1; zz<=reg[i].z+1; zz++) 
    for(yy=reg[i].y-1; yy<=reg[i].y+1; yy++)
    for(xx=reg[i].x-1; xx<=reg[i].x+1; xx++)
    {
      grads_az = angles->az->data[ zz + zsize*(xx + yy * xsize) ]; 
      grads_el = angles->el->data[ zz + zsize*(xx + yy * xsize) ]; 
      if( xx>=0 && yy>=0 && zz>=0 && 
          xx<xsize && yy<ysize && zz<zsize &&
          used->data[zz+(xx+yy*xsize)*zsize] != USED &&
          isaligned3(grads_az,grads_el,reg_az,reg_el,prec) )
      {
          /* add point */
          used->data[zz+(xx+yy*xsize)*zsize] = USED;
          reg[*reg_size].x = xx;
          reg[*reg_size].y = yy;
          reg[*reg_size].z = zz;
          ++(*reg_size);

          /* update region's angle */
          sincos(grads_az,&saz,&caz);
          sincos(grads_el,&sel,&cel);
          sumdx += caz*sel;
          sumdy += saz*sel;
          sumdz += cel;
          reg_az = atan2(sumdy,sumdx);
          reg_el = acos(sumdz/sqrt(sumdx*sumdx + sumdy*sumdy + sumdz*sumdz));
      }
    }
  (*reg_angle)->az=reg_az;
  (*reg_angle)->el=reg_el;
}

/*----------------------------------------------------------------------------*/
/** Build a 3D region of pixels that share the same angle, up to a
    tolerance 'prec', starting at point (x,y,z). 

    Alignment based on the initial angle of an input line (lstheta), 
    until the region becomes stable in the proper direction 
    Hence, alignment check becomes orthogonal, and NFA
    may be redone for an orthogonality check  
    
    Region growing is constrained to parallel 3D orientations, 
    since we would require the principal axis a-prior 
    to grow based on orthogonal orientation to said axis.   
*/
void region3_growORTH(int x, int y,int z, 
                         image3_double modgrad, grads angles, 
                         struct point3 * reg, int * reg_size, 
                         angles3 * reg_angle,  angles3 * lstheta, 
                         image3_char used,double prec ,int NOUT)
{
  double sumdx,sumdy,sumdz;
  int xx,yy,zz,i;

  /* check parameters */
  if( x < 0 || y < 0 || z<0 || x >= (int) used->xsize 
          || y >= (int) used->ysize || z >= (int) used->zsize)
    error("region3_grow: (x,y,z) out of the image.");
  if( angles->az == NULL || angles->az->data == NULL )
    error("region3_grow: invalid grads 'angles' or image 'az'.");
  if( reg == NULL ) error("region3_grow: invalid 'reg'.");
  if( reg_size == NULL ) error("region3_grow: invalid pointer 'reg_size'.");
  if( reg_angle == NULL ) error("region3_grow: invalid pointer 'reg_angle'.");
  if( lstheta == NULL ) error("region3_growORTH: invalid pointer 'lstheta'.");
  if( used == NULL || used->data == NULL )
    error("region3_grow: invalid image 'used'.");

  /* first point of the region */
  *reg_size = 1;
  reg[0].x = x;
  reg[0].y = y;
  reg[0].z = z;
  /*regions angles*/
  int xsize = (int)used->xsize;
  int ysize = (int)used->ysize;
  int zsize = (int)used->zsize;
  used->data[z+(x+y*xsize)*zsize] = USED;

  /* try neighbors as new region points */
  double weight, xcw, ycw, zcw, sum, xc, yc, zc;
  weight=0.; xcw=0.; ycw=0.; zcw=0.; sum=0.; xc=0.; yc=0.; zc=0.;
  weight = modgrad->data[z+(x+y*xsize)*zsize];
  xcw+=(double)x*weight;
  ycw+=(double)y*weight;
  zcw+=(double)z*weight;
  sum+=weight;
  xc = xcw/sum;
  yc = ycw/sum;
  zc = zcw/sum;
;
  angles3 regtheta = new_angles3((*lstheta)->az,(*lstheta)->el);
  double grads_az,grads_el;
  double prectemp = prec;
  int transition = 0;
  int gradalg;

  for(i=0; i<*reg_size; i++)
    for(zz=reg[i].z-1; zz<=reg[i].z+1; zz++) 
    for(yy=reg[i].y-1; yy<=reg[i].y+1; yy++)
    for(xx=reg[i].x-1; xx<=reg[i].x+1; xx++)
    {
      //check alignment with transition for apriori/aposteriori check
      grads_az = angles->az->data[ zz + zsize*(xx + yy * xsize) ]; 
      grads_el = angles->el->data[ zz + zsize*(xx + yy * xsize) ]; 
      if (((*reg_size)>8) && isaligned3((*lstheta)->az,(*lstheta)->el,regtheta->az,regtheta->el,prec))
        transition = 0;//DISABLED.
      if (transition==0) 
        gradalg = isaligned3ORTH(grads_az,grads_el,(*lstheta)->az,(*lstheta)->el,prec);
      else
        gradalg = isaligned3ORTH(grads_az,grads_el,regtheta->az,regtheta->el,prec);
      if( xx>=0 && yy>=0 && zz>=0 && 
          xx<xsize && yy<ysize && zz<zsize &&
          used->data[zz+(xx+yy*xsize)*zsize] != USED &&
          gradalg )
      {
          /* add point */
          used->data[zz+(xx+yy*xsize)*zsize] = USED;
          reg[*reg_size].x = xx;
          reg[*reg_size].y = yy;
          reg[*reg_size].z = zz;
          ++(*reg_size);

          /* update region's angle */
          weight = modgrad->data[zz+(xx+yy*xsize)*zsize];
          xcw+=(double)xx*weight;
          ycw+=(double)yy*weight;
          zcw+=(double)zz*weight;
          sum+=weight;
          xc = xcw/sum;
          yc = ycw/sum;
          zc = zcw/sum;
  	      regtheta = get_theta3(reg,*reg_size,xc,yc,zc,modgrad,*lstheta,prec,0);
      }
    }
  //UPDATE DISABLED
  (*reg_angle)->az = (*lstheta)->az;//regtheta->az;
  (*reg_angle)->el = (*lstheta)->el;//regtheta->el;
  free((void*)regtheta);
}


/*----------------------------------------------------------------------------*/
/** Helper function for trying variations to improve NFA value.
 */
double rect3_improve_update(struct rect3  r, grads angles,double logNT,int Nnfa,
                                 double* mnfa, double* mnfap, int minsize,
                                 double* mnfa_2,double* mnfap_2, int minsize2,
                                 double* mnfa_4,double* mnfap_4, int minsize4,
                                 double p1check, double p2check,
                                 struct rect3 * rec,double log_nfa,int orth)
{
  double log_nfa_new;
  /*Pick which NFA function to call based on sequence flag "orth" and the r.p value*/
  if (orth==0)
  {
    if(r.p>=p1check)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa,Nnfa,minsize);
    else if(r.p>=p2check)
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_2,Nnfa,minsize2);
    else
      log_nfa_new = rect3_nfa(&r,angles,logNT,mnfa_4,Nnfa,minsize4);
  }
  else if (orth==1)
  {
    if(r.p>=p1check)
      log_nfa_new = rect3_nfaORTH(&r,angles,logNT,mnfa,mnfap,Nnfa,minsize);
    else if(r.p>=p2check)
      log_nfa_new = rect3_nfaORTH(&r,angles,logNT,mnfa_2,mnfap_2,Nnfa,minsize2);
    else
      log_nfa_new = rect3_nfaORTH(&r,angles,logNT,mnfa_4,mnfap_4,Nnfa,minsize4);
  }
  else error("rect3_improve_update: post flag not recognized 0/1");

  /*Return original estimate or update rectangle appropriately*/
  if (log_nfa==-1)
    return log_nfa_new;
  if(log_nfa_new > log_nfa)
  {
    rect3_copy(&r,rec);
    return log_nfa_new;
  }
  else return log_nfa;
   
}

/*----------------------------------------------------------------------------*/
/** Try some 3D rectangular prism variations to improve NFA value. Only if the
    rectangle is not meaningful (i.e., log_nfa <= log_eps).

    For 3D analysis, we use geometric width series per Lezama (2015), 
    since regions of 2D surfaces can have very large widths compares to 
    the thin edge lines along the original 2D algorithm by VonGioi.  
    The azimuth/elevation widths are refined in alternating order. 
    
    We introduce an modified precision check for the MNFA tables to allow 
    for modified tolerances
 
 */
double rect3_improve( struct rect3 * rec, grads angles,
                            double logNT, double log_eps,
                            double* mnfa,double* mnfa_2,double* mnfa_4,
                            double*mnfap,double*mnfap_2,double*mnfap_4,
                            int Nnfa,int minsize, int minsize2,int minsize4,int orth)
{
  struct rect3 r;
  double log_nfa,log_nfa_new;
  log_nfa=-1;
  double factor = 1./sqrt(2.);
  //geometric width variation 
  double delta = (double)((int)max1(rec->width1,rec->width2)/5)/2.; 
  double delta_2 = delta / 2.0;
  int n;
  rect3_copy(rec,&r);
  //MNFA markov table flag for greater control over p settings
  double p1check=(double)r.p+.0001;//0.1;
  double p2check=((double)r.p/2)+.0001;//0.05;
  
  log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
		mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
    p1check,p2check,rec,log_nfa,orth);
  if( log_nfa > log_eps ) return log_nfa;

  /* try finer precisions, using correct mnfa table */
  rect3_copy(rec,&r);
  for(n=0; n<1; n++)
  {
    r.p /= 2.0;
    r.prec = r.p * M_PI;
    log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
		  mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
      p1check,p2check,rec,log_nfa,orth);
  }
  if( log_nfa > log_eps ) return log_nfa;
  
  /* try to reduce width, using geometric 1/sqrt(2) series
   * per Lezama 2015.  Alternate reducing both tangents */
  rect3_copy(rec,&r);
  for(n=0; n<5; n++)
  {
    if( (r.width1*factor) >= 0.5 )
    {
      r.width1 *=factor;
      log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
		    mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
        p1check,p2check,rec,log_nfa,orth);
    }

    if( (r.width2*factor) >= 0.5 )
    {
      r.width2 *=factor;
      log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
		    mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
        p1check,p2check,rec,log_nfa,orth);
    }
  }
  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce one side of the rectangle */
  rect3_copy(rec,&r);
  for(n=0; n<5; n++)
  {
    if( (r.width1*factor) >= 0.5 )
    {
      delta_2 = r.width1*(1.0-factor)/2.;
      r.x1 +=  r.dw1[0] * delta_2;
      r.y1 +=  r.dw1[1] * delta_2;
      r.z1 +=  r.dw1[2] * delta_2;
      r.x2 +=  r.dw1[0] * delta_2;
      r.y2 +=  r.dw1[1] * delta_2;
      r.z2 +=  r.dw1[2] * delta_2;
      r.width1 *= factor;
      log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
	    	mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
        p1check,p2check,rec,log_nfa,orth);
    }

    if( (r.width2*factor) >= 0.5 )
    {
      delta_2 = r.width2*(1.0-factor)/2.;
      r.x1 +=  r.dw2[0] * delta_2;
      r.y1 +=  r.dw2[1] * delta_2;
      r.z1 +=  r.dw2[2] * delta_2;
      r.x2 +=  r.dw2[0] * delta_2;
      r.y2 +=  r.dw2[1] * delta_2;
      r.z2 +=  r.dw2[2] * delta_2;
      r.width2 *= factor; 
      log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
		    mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
        p1check,p2check,rec,log_nfa,orth);
    }
  }
  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce the other side of the rectangle */
  rect3_copy(rec,&r);
  for(n=0; n<5; n++)
  {
    if( (r.width1*factor) >= 0.5 )
    {
      delta_2 = r.width1*(1.0-factor)/2.;
      r.x1 -=  r.dw1[0] * delta_2;
      r.y1 -=  r.dw1[1] * delta_2;
      r.z1 -=  r.dw1[2] * delta_2;
      r.x2 -=  r.dw1[0] * delta_2;
      r.y2 -=  r.dw1[1] * delta_2;
      r.z2 -=  r.dw1[2] * delta_2;
      r.width1 *=factor; 
      log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
		    mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
        p1check,p2check,rec,log_nfa,orth);
    }

    if( (r.width2*factor) >= 0.5 )
    {
      delta_2 = r.width2*(1.0-factor)/2.;
      r.x1 -=  r.dw2[0] * delta_2;
      r.y1 -=  r.dw2[1] * delta_2;
      r.z1 -=  r.dw2[2] * delta_2;
      r.x2 -=  r.dw2[0] * delta_2;
      r.y2 -=  r.dw2[1] * delta_2;
      r.z2 -=  r.dw2[2] * delta_2;
      r.width2 *= factor; 
      log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
		    mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
        p1check,p2check,rec,log_nfa,orth);
    }
  }
  if( log_nfa > log_eps ) return log_nfa;

  /* try even finer precisions */
  rect3_copy(rec,&r);
  for(n=0; n<1; n++)
  {
    r.p /= 2.0;
    r.prec = r.p * M_PI;
    log_nfa=rect3_improve_update(r,angles,logNT,Nnfa,
		  mnfa,mnfap,minsize,mnfa_2,mnfap_2,minsize2,mnfa_4,mnfap_4,minsize4,
      p1check,p2check,rec,log_nfa,orth);
  }
  return log_nfa;
}

/*----------------------------------------------------------------------------*/
/** Reduce the 3D region size, by elimination the points far from the
    starting point, until that leads to rectangle with the right
    density of region points or to discard the region if too small.
 */
int reduce_region3_radius( struct point3 * reg, int * reg_size,
                                 image3_double modgrad, angles3 reg_angle,
                                 double prec, double p, struct rect3 * rec,
                                 image3_char used, grads angles,
                                 double density_th , int orth)
{
  double density,rad1,rad2,rad,xc,yc,zc;
  int i;

  /* check parameters */
  if( reg == NULL ) error("reduce_region3_radius: invalid pointer 'reg'.");
  if( reg_size == NULL )
    error("reduce_region3_radius: invalid pointer 'reg_size'.");
  if( prec < 0.0 ) error("reduce_region3_radius: 'prec' must be positive.");
  if( rec == NULL ) error("reduce_region3_radius: invalid pointer 'rec'.");
  if( used == NULL || used->data == NULL )
    error("reduce_region3_radius: invalid image 'used'.");
  if( angles->az == NULL || angles->az->data == NULL )
    error("reduce_region3_radius: invalid image 'angles'.");

  /* compute region points density */
  /*require volume in denominator, in place of area*/
  density = (double) *reg_size /
                     ( dist3(rec->x1,rec->y1,rec->z1,rec->x2,rec->y2,rec->z2) 
                     * rec->width1 * rec->width2 );

  /* if the density criterion is satisfied there is nothing to do */
  if( density >= density_th ) return TRUE;

  /* compute region's radius: distance from centerpoint to furthest endpoint*/
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  zc = (double) reg[0].z;
  rad1 = dist3( xc, yc, zc, rec->x1, rec->y1, rec->z1 );
  rad2 = dist3( xc, yc, zc, rec->x2, rec->y2, rec->z2 );
  rad = rad1 > rad2 ? rad1 : rad2;

  /* while the density criterion is not satisfied, remove farther pixels */
  while( density < density_th )
  {
    rad *= 0.75; /* reduce region's radius to 75% of its value */

    /* remove points from the region and update 'used' map */
    for(i=0; i<*reg_size; i++)
      if( dist3( xc, yc, zc,  
        (double) reg[i].x, (double) reg[i].y, (double) reg[i].z ) > rad )
      {
        /* point not kept, mark it as NOTUSED */
        used->data[ reg[i].z+(reg[i].x+reg[i].y*used->xsize)*used->zsize ] = NOTUSED;
        /* remove point from the region */
        reg[i].x = reg[*reg_size-1].x; /* if i==*reg_size-1 copy itself */
        reg[i].y = reg[*reg_size-1].y;
        reg[i].z = reg[*reg_size-1].z;
        --(*reg_size);
        --i; /* to avoid skipping one point */
      }

    /* reject if the region is too small.
       2 is the minimal region size for 'region2rect' to work. */
    if( *reg_size < 2 ) return FALSE;

    /* re-compute rectangle */
    region2rect3(reg,*reg_size,modgrad,reg_angle,prec,p,rec,orth);

    /* re-compute region points density */
    density = (double) *reg_size /
                       ( dist3(rec->x1,rec->y1,rec->z1,rec->x2,rec->y2,rec->z2) 
                       * rec->width1 * rec->width2 );
  }

  /* if this point is reached, the density criterion is satisfied */
  return TRUE;
}

/*----------------------------------------------------------------------------*/
/** Refine a 3D rectangular prism .
 
    Per Liu's LSDSAR, we drop the adaptive tolerance estimation, 
    since we are using explicitly defined Markov transition matrices
    which are dependent upon the tolerance parameter.   

    Originally: an estimation of the angle tolerance is performed by the
    standard deviation of the angle at points near the region's
    starting point. Then, a new region is grown starting from the same
    point, but using the estimated angle tolerance. If this fails to
    produce a rectangle with the right density of region points,
    'reduce_region_radius' is called to try to satisfy this condition.
 */
int refine3( struct point3 * reg, int * reg_size, image3_double modgrad,
                   angles3 reg_angle, double prec, double p, struct rect3 * rec,
                   image3_char used, grads angles,
                   double density_th , int NOUT, int orth)
{

  double tau,density;
  int i;//,n;

  /* check parameters */
  if( reg == NULL ) error("refine3: invalid pointer 'reg'.");
  if( reg_size == NULL ) error("refine3: invalid pointer 'reg_size'.");
  if( prec < 0.0 ) error("refine3: 'prec' must be positive.");
  if( rec == NULL ) error("refine3: invalid pointer 'rec'.");
  if( used == NULL || used->data == NULL )
    error("refine3: invalid image 'used'.");
  if( angles ==NULL || angles->az == NULL || angles->az->data == NULL )
    error("refine3: invalid image 'angles'.");
  if((orth!=1)&&(orth!=0)) error("refine3: orth flag not 0/1");

  /* compute region points density */
  density = (double) *reg_size /
                     ( dist3(rec->x1,rec->y1,rec->z1,rec->x2,rec->y2,rec->z2) 
                     * rec->width1 * rec->width2 );

  /* if the density criterion is satisfied there is nothing to do */
  if( density >= density_th ) return TRUE;

  /*------ First try: reduce angle tolerance ------*/

  int xsize = (int) used->xsize;
  int zsize = (int) used->zsize;
  for(i=0; i<*reg_size; i++) //reset exclusion principal
    used->data[ reg[i].z + (reg[i].x + reg[i].y * xsize) * zsize ] = NOTUSED;
  /*STANDARD DEVIATION ESTIMATE IGNORED PER LIU ET AL*/
  /* find a new region from the same starting point and new angle tolerance */
  tau=prec/2.0;
  if(orth==0) 
    region3_grow(reg[0].x,reg[0].y,reg[0].z,angles,reg,reg_size,&reg_angle,used,tau,NOUT);
  else
    region3_growORTH(reg[0].x,reg[0].y,reg[0].z,modgrad,angles,reg,reg_size,&reg_angle,&reg_angle,used,tau,NOUT);
  
  p=tau/M_PI;
  /* if the region is too small, reject */
  if( *reg_size < 2 ) return FALSE;

  /* re-compute rectangle */
  region2rect3(reg,*reg_size,modgrad,reg_angle,tau,p,rec,orth);

  /* re-compute region points density */
  density = (double) *reg_size /
                     ( dist3(rec->x1,rec->y1,rec->z1,rec->x2,rec->y2,rec->z2) 
                     * rec->width1 * rec->width2 );
                     
  /*------ Second try: reduce region radius ------*/
  if( density < density_th )
    return reduce_region3_radius( reg, reg_size, modgrad, reg_angle, prec, p,
                                 rec, used, angles, density_th ,orth);

  /* if this point is reached, the density criterion is satisfied */
  return TRUE;
}



