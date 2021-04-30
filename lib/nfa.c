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
#include <stdlib.h>
#include <math.h>
#include <tgmath.h>
#include <limits.h>
#include <float.h>
#include<string.h>
#include <time.h>
#include<gsl/gsl_eigen.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_cdf.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_qrng.h>
#include<gsl/gsl_sf_gamma.h>
#include<gsl/gsl_sf_trig.h>
#include<sys/mman.h>

#include "nfa.h"
#include "constants.h"
#include "misc.h"
#include "tuples.h"

/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/


/*NOTE: Here cut vonGioi's log functions, unused in LSDSAR in favor of Markov table*/

/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).

    NFA stands for Number of False Alarms:
    @f[
        \mathrm{NFA} = NT \cdot B(n,k,p)
    @f]

    - NT       - number of tests
    - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
    @f[
        B(n,k,p) = \sum_{j=k}^n
                   \left(\begin{array}{c}n\\j\end{array}\right)
                   p^{j} (1-p)^{n-j}% alpha=2;
*/
double nfa(int n, int k, double p, double logNT,double *mnfa,int N)
{
   if(n>N||k>N)
        return 101;
  /* check parameters */
  if( n<0 || k<0 || k>n || p<=0.0 || p>=1.0 )
    error("nfa: wrong n, k or p values.");
  /* trivial cases */
  if( n<3 || k==0 ) return -logNT;
  return mnfa[k*N+n]-logNT;
}


/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).
 * 
 *  Use a negative binomial approximation as formulated by 
 *  Xia, A. & Zhang, M. (2010). "On approximation of Markov binomial distributions".  
 *    Bernoulli 15(4), 2009, 1335-1350: DOI: 10.3150/09-BEJ194
*/
double nfaORTH(int n, int k, double pp, double logNT, double *mnfa, int N)
{ 

  //let *mfa store :
  //     mnfa = [p0,p11,p01] for p for mfa, mnfa_2, mnfa_4     
  int nscale = 1;
  //if((n>N)||(k>N))
  //     return 101;
  /* check parameters */
  if( n<0 || k<0 || k>n || pp<=0.0 || pp>=1.0 )
    error("nfa: wrong n, k or p values.");
  /* trivial cases */
  if( n<3 || k==0 ) return -logNT;

  double p0  = mnfa[0];
  double p11 = mnfa[1];
  double p01 = mnfa[2];
  double p1=1.0-p0;
  double p10=1.0-p11;
  double p00=1.0-p01;
  double *plk0;

  double alpha = p01; 
  double beta = p11;
  double p  = alpha/(1.-beta+alpha);
  double pi = (1.-beta)/(1.-beta+alpha);

  double A0 = (2.*alpha*(1.-beta)*(beta-alpha))/pow(1.-beta+alpha,3.);
  double tempp2=pow(p,2.);
  double A1 = (2.*alpha*(1.-beta)*(beta-alpha))/pow(1.-beta+alpha,4.);
  double nn=(double) n;
  double ES   = nn*p;
  double VarS = nn*p*(1.-p)+nn*A0-A1+A1*pow(beta-alpha,nn);
  
  double output, div;
  double r, q, mhat, m, theta;
  
  double mu1 = (1.-alpha)/alpha;
  double sig1= (1.-alpha)/pow(alpha,2.);
  double mu2 = beta/(1.-beta);
  double sig2= beta/pow(1.-beta,2.);
  double C0 = fabs(beta-alpha)*(5.+max2(43.*alpha,beta))/pow(1.-max2(beta,alpha),2.);
  double C1 = 10.*max2(beta,alpha)/(1.-max2(beta,alpha));
  double C2 = (1.-p)*(5.+max2(23.*alpha,beta))/pow(1.-max2(alpha,beta),2.);
  double K1 = sqrt(5.)*sqrt((mu1+mu2+2.)/min2(min2(1.-alpha,beta),0.5));
  double K2 = 90.*(sig1+sig2)/(mu1+mu2+2.);
  
  double tempoutput=0;
  double temppost=0;
  if (VarS>=ES)  
  {
    r=pow(ES,2.) / (VarS-ES) ; 
    q=ES/VarS;
    temppost=pow(beta,floor(nn/4.));
    div = C0*(2.*K1/sqrt(nn)+4.*K2/nn + temppost);//pow(beta,floor(nn/4.)));
    /*output = 1 - NB(r,q) //negative binomial 
      gsl: (ushort k, p, n )
      match: p^n (1-p)^k = q^r (1-q)^k ; so (k,p,n)=(k,q,r) */
    tempoutput = gsl_cdf_negative_binomial_Q((unsigned int)k,(double)q,(double)r);//ushort k | p, n
    tempoutput=-log10(tempoutput)-logNT;
    if(!isfinite(tempoutput))
    {
      double to1a,to1b,to1c,to1,to2,to3;
      to1a= gsl_sf_lngamma((double)r+(double)k);
      to1b= -gsl_sf_lngamma((double)k+1.);//log10l(tgammal(kl+1.));
      to1c= -gsl_sf_lngamma((double)r);//log10l(tgammal(nl));
      to1 = to1a+to1b+to1c;
      to2 = log(q)*r ;
      to3 = log(1.-q)*(double)k; 
      tempoutput=(to1+to2+to3);// @p
      double tempoutput2;
      for(int kk=k+1;k<floor(r);k++)
      {
        to1a= gsl_sf_lngamma((double)r+(double)kk);
        to1b= -gsl_sf_lngamma((double)kk+1.);//log10l(tgammal(kl+1.));
        to1c= -gsl_sf_lngamma((double)r);//log10l(tgammal(nl));
        to1 = to1a+to1b+to1c;
        //CONST:to2=-log10(q)*r ;
        to3=log(1.-q)*(double)kk; 
        tempoutput2=(to1+to2+to3);// @p
        //both temps are in -lot10 currently, convert back to basen before operation 
        tempoutput+=log1p(exp(tempoutput2-tempoutput));
      }  
      tempoutput/=-log(10);
      tempoutput-=logNT;
    }
  }
  else
  {
    mhat = pow(ES,2.)/(ES-VarS);
    m = floor(mhat);
    theta = nn*p/m;
    //div = (C1*fabs(p-theta)/(1.-theta) + C2*fabs(beta-alpha)/(1.-theta))*(  2.*K1/sqrt(nn)+4.*K2/nn + pow(max2(beta,alpha),floor(n/4.))   )+(pow(theta,2.)*(mhat-m)/(nn*p*(1.-theta)));
    tempoutput = gsl_cdf_binomial_Q((unsigned int)k,(double)theta,(int)mhat);//ushort k | p, n
    tempoutput=-log10(tempoutput)-logNT;

  }
  output = (double) tempoutput;
  //printf("\tdivergence: %.2e, nlog-divergence: %.2f\n",div,nlogdiv);fflush(stdout);
  //printf("\t n %d divergence %.8f ",n,div);fflush(stdout);
  return output;
}

