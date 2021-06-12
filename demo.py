'''
PyCIS - Python Computational Inference from Structure

    A-contrario inference of object trajectories from structure-in-noise, 
    building on Line Segment Detection (LSD) for dense electro-optical time-series data
    formatted as 3D data cubes, with markov kernel estimation for non-uniform noise models.
    LSD C-extension module equipped with multi-layer a-contrario inference for center-line features
    from gradient information.  Python modules provided for inference of feature classifications
    using second-order gestalts, and ingesting/plotting of FITS-format data files.

    This software is under active development.

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
*Date of Modification: June 07, 2021

**NOTICE: For copyright and licensing, see 'notices' at bottom of README
'''

## IMPORT NECESSARY LIBRARIES
import faulthandler; faulthandler.enable()
import pycis
import os
import time
import numpy as np
# Import other python functions 
from pylib.import_fits import import_fits, mod_fits
from pylib.detect_outliers import detect_outliers
from pylib.print_detections import print_detections
from pylib.run_astrometry import run_astrometry

def get_size(I):
    '''    Get matrix size variables with necesary 0-setting    '''
    M = len(I)
    if (I.ndim==1):
        M=len(I)
        N=0
        O=0
    elif (I.ndim==2) and (M>10):
        M,N=np.shape(I)
        O=0
    elif (I.ndim==3) and (M>10):
        M,N,O=np.shape(I)
    else:
        M,N,O=0,0,0
    return M,N,O

def get_sizes(I,I0):
    '''    Get matrix size variables with necessary 0-settings    '''
    M,N,O = get_size(I)
    print('M %d, N %d, O %d'%(M,N,O))
    M0,N0,O0 = get_size(I0)
    print('M0 %d, N0 %d, O0 %d'%(M0,N0,O0))
    return M,N,O,M0,N0,O0

def my_flatten(I,M,N,O):
    '''    Format image data for passing to pycis    '''
    if (O==0):
        I = I.T.flatten().tolist()
    if (O>0):
        templist = np.zeros((M*N*O,))
        for i in range(M):
            for j in range(N):
                for k in range(O):
                    templist[k+O*(i+M*j)] = I[i,j,k]
        I = templist.tolist()
    return I

def main(I,I0, folder='results',name='temp',
    a=4.,d=.4,t=1.,p=[0,0,0,0,0,0],p2=[0,0,0,0,0,0],getp=1,scale=0.8,sigma=0.6,e=0):
    '''
    Run LSD pipeline for a test and conditioning image pair .
    Reshape data and set input parameter vector.
    Decide between output functions.
    
    Input:  
        I :     test image
        I0:     noise model (or prior edge lines)
        folder: location for results 
        name:   name for saving .png and .npy files
        a:      gradient-by-ratio parameter.  Produces a (k*2+1)^d kernel for k=log(10)*a
                    eg a=1 produces a 7^d kernel, a=2 an 11^d kernel, a=3 a 15^d kernel, etc
        d:      density threshold for improving regions.  Higher values enforce stronger linearity constraints   
        t:      denominator factor for tolerance.  Initial tolerance is 22.5deg, empirically optimal for 2D
        p:      markov kernel for estimating parallel alignments 
                    (in edge case, build regions by parallel alignment and count alignments )
        p2:     markov kernel for estimating parallel alignments 
                    (in edge case, build regions by parallel alignment and count alignments )
        getp:   flag for pipline - control output as markov kernel or lines.  see pycis.c
        scale:  fraction of gaussian downsampled side lengths to input data (set to 1 for no downsampling)
        sigma:  std=sigma/scale for variance of gaussian downsampling via seperable 1D kernels
        e:      -log10(epsilon) NFA threshold.  Choose 0 for epsilon=1 (default theory).  
                    Robust to selection, but setting very large (e=6) can reduce spurious noise
                    and improve 2nd-order detection if there exists a statistically relevant number of detections
           
    Output: one of the following: 
        inputv/inputvorth:  updated settings vector with solved markov kernels
        data1_name.npy:     edge line detections
        data2_name.npy:     center line detections

    Notes: I,I0 cannot be empty vectors.  To set an 'empty image',
           use a small matrix, e.g. 2x2 identity, and the function
           will set the variables X,X0 appropriatly.
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    ## SET INPUT DIRECTIONS FOR VERSIONING - LSD CANNOT ACCEPT EMPTY IMAGES, BUT REQUIRES M/M0=0
    M,N,O,M0,N0,O0 = get_sizes(I,I0)
  
    ## FLATTEN IMAGES FOR USE IN C, COPY FOR PLOTTING 
    I_full = np.copy(I)
    I0_full= np.copy(I0)
    I = my_flatten(I,M,N,O)
    I0= my_flatten(I0,M0,N0,O0)
    
    ## SET INPUT PARAMETER VECTOR 
    p11= p[0];p01= p[1];p11_2= p[2];p01_2= p[3];p11_4= p[4];p01_4= p[5]
    dp11=p2[0];dp01=p2[1];dp11_2=p2[2];dp01_2=p2[3];dp11_4=p2[4];dp01_4=p2[5]
    alpha=a#4.
    eps=10.**(-1.*e)#(1/1.)
    density=d #0.4
    angth=22.5/t
    sizenum=np.sqrt(M**2.+N**2.)*5.
    if sizenum>(10.**4):
        sizenum=10.**4
    if O>0:
        sizenum=min(10.**6.,np.sqrt(M**2.+N**2.+O**2.)*(5.))
    inputv=[alpha,eps,density,sizenum,angth,
        p11, p01, p11_2, p01_2, p11_4, p01_4,
        scale,sigma]
    inputvorth=[alpha,eps,density,sizenum,angth,
        dp11,dp01,dp11_2,dp01_2,dp11_4,dp01_4,
        scale,sigma]
   
    ## RUN LSD
    #Get markov kernel if requested
    markov = getp
    if markov>1:
        if markov==2:
            print('\n-------------------- MARKOV-PARALLEL: %s --------------------\n'%name,flush=True)  
        else:
            print('\n-------------------- MARKOV-ORTHOGONAL: %s --------------------\n'%name,flush=True)  
        time.sleep(1)
        lines = pycis.pycis(I,M,N,O,I0,M0,N0,O0,inputv,inputvorth,markov)
        del I, I0, alpha, density, angth, a,d,t,inputv,I_full,I0_full
        #Return inputv 
        return lines    
    else:
        #Find kernel and run estimation...
        savename='%s/data1_%s'%(folder,name)
        if markov==1: #markov = 1, run markov estimation plus edge detection 
            print('\n-------------------- LSD + Markov: %s --------------------\n'%name,flush=True)    
        elif markov==0 and M0==0: #markov=0, edges with prior markov
            print('\n-------------------- LSD (Edge): %s --------------------\n'%name,flush=True)    
        elif markov==0 and M0>0: #markov=0, centerlines with prior markov
            print('\n-------------------- LSD (Centers): %s --------------------\n'%name,flush=True)   
            savename = '%s/data2_%s'%(folder,name) 
        else:
            print("ERROR: incompatible markov and X0 input .",flush=True)
            quit()

        time.sleep(1)
        lines = pycis.pycis(I,M,N,O,I0,M0,N0,O0,inputv,inputvorth,markov)
        #lines = [];

        ## SAVE LINE RESULTS 
        np.save(savename,lines)
      
        del I, I0, alpha,density,angth, a,d,t,inputv,I_full,I0_full
        #Return nothing - data is saved to file
        return lines

if __name__=="__main__":
    '''
    Main interface : run test 
    Input: data/...
        yyymmdd_norad_satname/*.fit - a folder with raw fits frames
    Output: results/... 
        data1_name.npy - edge line detections
        data2_name.npy - center line detections
        goodlines_name.npy - 2nd-order meaningful detections
        badlines_name.npy - 1st-order meaningful only detections
        img_name.npy - last fit frame data with projected detections (2nd order in red)
        vidAll_name.npy - animated fits frame data with projected detections over time (2nd order in red)
        vidObj_name.npy - animated fits frame data with only 2nd-order projected detections over time 
    '''

    ## BEGIN TIMING 
    starttime=time.time()
    numtests =1

    ## LOAD IMAGES 
    satfolder='data'
    satlist = ['20201220_45696_starlink-1422',]
    outfolder='results'
    ## IF NOT ON TACC, USE THE SETTINGS IN FOLLOWING COMMENTS
    framerange=[8,33] #[8,17]
    scale = 1 #0.1

    #Select data from folder - loopable 
    for satname in satlist:
        prename='{}'.format(satname)
        
        #import data and reshape as needed
        input_dir = '%s/%s'%(satfolder,satname)
        I3, headers = import_fits(input_dir, savedata=0, subsample=1, 
            framerange=framerange, scale=scale)
        #I3 = np.load('%s/%s.npy'%(satfolder,satname))
        I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))
        #I03 = np.copy(I3)     
        emptyset = np.eye(2)


        #LSD settings 
        a = 1.0  #Gradient-by-Ratio parameter 
        t = 1.0  #Division factor for angular tolerance
        d = 0.2 #Density threshold 
        scale=1. #Gaussian downsampling volume ratio 
        sigma=.6 #Gaussian downsampling deviation factor
        e=0      #NFA epsilon parameter
        #Fix name
        #name = '%s_a%.2f_t%.2f_d%.2f_e%.2f_sc%.2f_si%.2f'%(prename,
        #    a,t,d,e,scale,sigma)
        name = prename
        
         

        #Get parallel markov kernel data 
        getp = 2 
        outv = main(np.copy(emptyset),np.copy(I3),
            folder=outfolder,name=name,
            a=a,d=d,t=t,scale=scale,sigma=sigma, e=e,
            getp=getp)
        p=np.copy(outv[5:11])

        #Get orthogonal markov kernel data 
        getp = 3
        outv = main(np.copy(emptyset),np.copy(I3),
            folder=outfolder,name=name,
            a=a,d=d,t=t,scale=scale,sigma=sigma, e=e,
            getp=getp)
        p2=np.copy(outv[5:11])

        ## GET LINE OUTPUT (SAVES LINE DATA )
        #   1) getp=1, use naive image to get markov kernel and detect edge lines
        #   2) getp=0, X0=0 (no data), use existing markov and detect edge lines 
        #   3) getp=0, X0>0 (line data), use existing markov and detect center lines 
        #       (for 3D input only, ie Z>0)
        #   
        '''
        if os.path.exists('%s/%s.npy'%(outfolder,name)):
            print("SKIPPING %s"%name,flush=True)
            continue 
        '''
        #Get edge lines
        getp = 0 
        lines = main(np.copy(I3),np.copy(emptyset),
            folder=outfolder,name=name,
            a=a,d=d,t=t,scale=scale,sigma=sigma, e=e,
            getp=getp, p=p, p2=p2)
        #Get center lines
        getp=0
        lines = np.array(lines).squeeze()
        lines = main(np.copy(I3),np.copy(lines),
            folder=outfolder,name=name,
            a=a,d=d,t=t,scale=scale,sigma=sigma, e=e,
            getp=getp, p=p, p2=p2)
    


        ## SECOND NFA APPLICATION (SAVES LINE DATA)
        goodlines, badlines = detect_outliers(np.copy(I3),lines,folder=outfolder,savename=name)

        ## CONSTRUCT VIDEO DATA OUTPUT
        print_detections(np.copy(I3),goodlines,badlines,folder=outfolder,savename=name)

        ## RUN ASTROMETRY AND UPDATE HEADERS
        headersnew = run_astrometry(goodlines, badlines, headers, folder=outfolder,savename=name)
        newfits = 'new%s/%s'%(satfolder,satname)
        if not os.path.exists('new%s'%satfolder):
            os.makedirs('new%s'%satfolder)
        if not os.path.exists(newfits):
            os.makedirs(newfits)
        mod_fits(input_dir, headersnew, folder=newfits,
            subsample=1, framerange=framerange)

    ## PLOT TIMING DATA 
    avgtime = (time.time()-starttime)/1. #numtests
    print('TOTAL RUNTIME: %.2f sec (%.2f min)\n\n'%(avgtime,avgtime/60.))

    
