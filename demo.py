'''
PyCIS - Python Computational Inference from Structure

demo_new.py: Main interface to PyCIS, with examples

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
*Date of Modification: December 17, 2021

**NOTICE: For acknowledgements, copyright, licensing see 'notices' in README
'''

## IMPORT NECESSARY LIBRARIES
import faulthandler; faulthandler.enable()
import os
import time
from pylib.main_pipe import run_pycis


def astrianavstardemo():
    '''    Demo of GEO navstar observation, tighter angular tolerance for detection   '''
    #SPECIFY DATA INPUT AND FOLDERS
    satfolder='data' #folder of all data
    satlist = ['20201224_26407_navstar-48',] #folder of test data
    datatype='fits' #test data suffix
    imgfolder='results_NavstarDemo' #where to store results
    framerange = [-1,18] #frames of test data to use, default all [-1,-1]
    imscale = 1 #scale of test data to import (crop precentage)
    numsteps = 5 #number of partitioning parallelization steps (numsteps-x-numsteps partitioning)
 
    #PRINTING OPTIONS   
    printonly = 0 #flag to only print input data and exit (initial visualization)
    makeimg = 1 #flag to print still images
    makegif = 1 #flag to print animations
    printcluster=0 #flag to print 2nd order clustering
    vs = 0.25 #scale of printing (precentage)
    fps=5 #fps of animation .gif
    

    #PIPELINE FLOW OPTIONS
    solvemarkov=0 #flag solve local markov kernels or assume global uniformity
    resolve=0 #flag to enable/disable resolving of 1st-order line detections
    cluster = 2 #flag to disable(0)/ resolve (1)/ or use existing (2) 2nd-order clustering 
    solveastro=0 #flag to disable(0)/ resolve (1)/ or use existing (2) astrometry solution
    imgastro=1 #flag to use image data in ranking star values (1) or rank by NFA (0)
    tle=[] #optional TLE to use, if present generates an 'expected track' for precision-recall analysis

    #HYPERPARAMETERS
    a=0.0 #gradient kernel for 1st-order line detection.  3x3 sobel kernel (0) or radius of GR kernel.
    t=1.3 #tightening factor for 1st-order angular tolerance (threshold tau/t)
    median = 0 #flag to subtract the median value of each pixel, e.g. to remove hot pixels
    binfactor=1 #integer binning factor 

    #OTHER PARAMETERS
    e2 = 0 #meaninfulness threshold of clustering algorithm, used in precision-    recall analysis
    shift=0 #artifical suppression of tracking error by shifting FITS frames

    #RUN PYCIS
    tlist = [] #record runtime for multiple hyperparameter options
    for dummy_parameter in [0,]: #may iterate over several hyperparameter options 
        imgname = 'a%dt%dm%db%d'%(a,int(t*10),median,binfactor) #image name to save, listing hyperparameter options
        linename = '%s_e%d'%(imgname,int(e2*100)) #For PR analysis, can fix 1st-order detections and redo clustering for e2 options
        stime = time.time()     
        run_pycis(
            satfolder,satlist,datatype,numsteps,
            imgfolder,imgname,vs,makegif,printcluster,
            solvemarkov,resolve,printonly,
            imscale,framerange,a,t,median,shift,e2=e2,makeimg=makeimg,
            linename=linename,binfactor=binfactor,fps=fps,tle=tle,
            imgastro=imgastro,cluster=cluster,solveastro=solveastro)
        tlist.append(time.time() - stime)
    print('TIME:')
    print(tlist)

def astriademo():
    '''    Demo of LEO starlink observation, binning and median subtraction    '''
     #SPECIFY DATA INPUT AND FOLDERS
    satfolder='data' #folder of all data
    satlist = ['20201220_45696_starlink-1422',] #folder of test data
    datatype='fits' #test data suffix
    imgfolder='results_StarlinkDemo' #where to store results
    framerange = [8,33] #frames of test data to use, default all [-1,-1]
    imscale = 1 #scale of test data to import (crop precentage)
    numsteps = 5 #number of partitioning parallelization steps (numsteps-x-numsteps partitioning)
 
    #PRINTING OPTIONS   
    printonly = 0 #flag to only print input data and exit (initial visualization)
    makeimg = 1 #flag to print still images
    makegif = 1 #flag to print animations
    printcluster=0 #flag to print 2nd order clustering
    vs = 0.25 #scale of printing (precentage)
    fps=5 #fps of animation .gif
    

    #PIPELINE FLOW OPTIONS
    solvemarkov=0 #flag solve local markov kernels or assume global uniformity
    resolve=0 #flag to enable/disable resolving of 1st-order line detections
    cluster = 2 #flag to disable(0)/ resolve (1)/ or use existing (2) 2nd-order clustering 
    solveastro=0 #flag to disable(0)/ resolve (1)/ or use existing (2) astrometry solution
    imgastro=1 #flag to use image data in ranking star values (1) or rank by NFA (0)
    tle=[] #optional TLE to use, if present generates an 'expected track' for precision-recall analysis

    #HYPERPARAMETERS
    a=0.0 #gradient kernel for 1st-order line detection.  3x3 sobel kernel (0) or radius of GR kernel.
    t=1.0 #tightening factor for 1st-order angular tolerance (threshold tau/t)
    median = 1 #flag to subtract the median value of each pixel, e.g. to remove hot pixels
    binfactor=2 #integer binning factor 

    #OTHER PARAMETERS
    e2 = 0 #meaninfulness threshold of clustering algorithm, used in precision-    recall analysis
    shift=0 #artifical suppression of tracking error by shifting FITS frames

    #RUN PYCIS
    tlist = [] #record runtime for multiple hyperparameter options
    for dummy_parameter in [0,]: #may iterate over several hyperparameter options 
        imgname = 'a%dt%dm%db%d'%(a,int(t*10),median,binfactor) #image name to save, listing hyperparameter options
        linename = '%s_e%d'%(imgname,int(e2*100)) #For PR analysis, can fix 1st-order detections and redo clustering for e2 options
        stime = time.time()     
        run_pycis(
            satfolder,satlist,datatype,numsteps,
            imgfolder,imgname,vs*float(binfactor),makegif,printcluster,
            solvemarkov,resolve,printonly,
            imscale,framerange,a,t,median,shift,e2=e2,makeimg=makeimg,
            linename=linename,binfactor=binfactor,fps=fps,tle=tle,
            imgastro=imgastro,cluster=cluster,solveastro=solveastro)
        tlist.append(time.time() - stime)
    print('TIME:')
    print(tlist)

if __name__=="__main__":
    '''   
    Run example demo scripts.  Be sure to download input ASTRIANet data according to README.
    '''
    astrianavstardemo()
    astriademo()  