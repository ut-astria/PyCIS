'''
PyCIS - Python Computational Inference from Structure

demo_new.py: Main interface to PyCIS, with examples

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
Date of Modification: May 5, 2022

--------------------------------------------------------------------
PyCIS: An a-contrario detection algorithm for space object tracking from optical time-series telescope data. 
Copyright (C) 2022, Benjamin G. Feuge-Miller, <benjamin.g.miller@utexas.edu>

This program is free software; you can redistribute it and/or modify
t under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
--------------------------------------------------------------------
'''

## IMPORT NECESSARY LIBRARIES
import faulthandler; faulthandler.enable()
import argparse
import time
import glob
import numpy as np
from astropy.io import fits
from pylib.main_pipe import run_pycis, end_pipe

def getsettings(satfolder,satlist,width=20,overlap=0):
    #Fetch list of files
    input_dir='%s/%s'%(satfolder,satlist)
    datatype='fit'
    imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
    listlen = len(imlist)
    if listlen==0:
        datatype='fits'
        imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
        listlen = len(imlist)
        if listlen==0: 
            print('NO FIT OR FITS FILES...')
            #quit()
            datatype='fit.zip'
            imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
            listlen = len(imlist)
            if listlen==0: 
                print('NO FIT.ZIP FILES, CHANGE DATATYPE')
                quit()
    #Fetch data info
    with fits.open(imlist[0]) as hdul:
        hdr=hdul[0].header
        #hdr will be leveraged more extensivly during TDM writing
        #tle = [hdr['TLELN1'],hdr['TLELN2']]
        exposure = hdr['EXPOSURE']
    #Choose pipeline settings 
    if exposure==0.1:
        print('ACTIVATING LEO SETTINGS FOR EXPOSURE=',exposure)
        t=1.0
        median=1
        binfactor=2
        runoutlierpair=[1,0]
        #outlier detection is for each block, large starfield rotation
    elif exposure==3.0:
        print('ACTIVATING GEO SETTINGS FOR EXPOSURE=',exposure)
        t=1.2#1.3
        median=1
        binfactor=1
        runoutlierpair=[0,1]
        #outlier detection is for all blocks, negligable starfield rotation
    else:
        print('ERROR: EXPOSURE=',exposure)
        print('\t Expected exposure of 0.1 (LEO) or 3.0 (GEO).  Aborting...')
        quit()
    #get winlist, a list of frameranges 
    print('\t [t, median, bin] = ',[t,median,binfactor])
    window=min(width,listlen) #3d sliding window
    winlist=[]
    init = [0,window-1]
    winlist.append(init)
    last=0
    while (last+window)<listlen: #last+2*window to avoid partial window
        #TODO: Confirm this works when we enable overlapping frames
        last = winlist[-1]
        last = last[-1] - overlap
        now = [last+1,last+window]
        if last+window<listlen:
            winlist.append(now)
        else:
            break
   

    return winlist, t, median, binfactor, runoutlierpair

def main(satfolder,satlist, framerange, t, median, binfactor,bigparallel,subprocess_count,injectScale=0,runoutlierpair=[0,1],frameidx=0,medianimg=None,minimg=0,p1=None,p2=None):
    '''    Demo of GEO navstar observation, tighter angular tolerance for detection   '''
    #SPECIFY DATA INPUT AND FOLDERS
    #satfolder= 'data' #folder of all data
    #satlist = ['20201224_26407_navstar-48',] #folder of test data
    datatype='fits' #test data suffix
    #datatype='fit.zip' #test data suffix
    #imgfolder='results_NavstarDemo' #where to store results
    #framerange = [-1,18] #frames of test data to use, default all [-1,-1]
    imscale = 1 #scale of test data to import (crop precentage)
    numsteps = 5 #number of partitioning parallelization steps (numsteps-x-numsteps partitioning)
   # if binfactor==1:
   #     numsteps=9
 
    #PRINTING OPTIONS   
    printonly = 0 #flag to only print input data and exit (initial visualization)
    makeimg = 0 #flag to print still images
    makegif = 0 #flag to print animations
    printcluster=0 #flag to print 2nd order clustering
    vs = 0.25 #scale of printing (precentage)
    fps=5 #fps of animation .gif
    

    #PIPELINE FLOW OPTIONS
    runoutlier=runoutlierpair[0]
    solvemarkov=0 #flag solve local markov kernels or assume global uniformity
    resolve = 1 #flag to enable/disable resolving of 1st-order line detections
    cluster = 1 #flag to disable(0)/ resolve (1)/ or use existing (2) 2nd-order clustering 
    solveastro=0 #flag to disable(0)/ resolve (1)/ or use existing (2) astrometry solution
    imgastro=1 #flag to use image data in ranking star values (1) or rank by NFA (0)
    tle=[] #optional TLE to use, if present generates an 'expected track' for precision-recall analysis

    #HYPERPARAMETERS
    a=0.0 #gradient kernel for 1st-order line detection.  3x3 sobel kernel (0) or radius of GR kernel.
    #t=1.3 #tightening factor for 1st-order angular tolerance (threshold tau/t)
    #median = 0 #flag to subtract the median value of each pixel, e.g. to remove hot pixels
    #binfactor=1 #integer binning factor 

    #OTHER PARAMETERS
    e2 = 0 #meaninfulness threshold of clustering algorithm, used in precision-    recall analysis

    #RUN PYCIS
    tlist = [] #record runtime for multiple hyperparameter options
    getmedianimg = 2 if frameidx>0 else 1
    for dummy_parameter in [0,]: #may iterate over several hyperparameter options 
        rangetag = 'A%dB%d'%(framerange[0]+1,framerange[1]+1)
        imgname = 'a%dt%dm%db%d_%s'%(a,int(t*10),median,binfactor,rangetag) #image name to save, listing hyperparameter options
        linename = '%s_e%d_inj%d'%(imgname,int(e2*100),injectScale) #For PR analysis, can fix 1st-order detections and redo clustering for e2 options
        #linename = '%s_e%d_inj%d'%(imgname,int(e2*100),injectScale) #For PR analysis, can fix 1st-order detections and redo clustering for e2 options
        stime = time.time()     
        output1,output2,outputP1,outputP2=run_pycis(
            satfolder,satlist,datatype,numsteps,
            imgfolder,tdmfolder,imgname,vs*float(binfactor),makegif,printcluster,
            solvemarkov,resolve,printonly,
            imscale,framerange,a,t,median,0,e2=e2,makeimg=makeimg,
            linename=linename,binfactor=binfactor,fps=fps,tle=tle,
            imgastro=imgastro,cluster=cluster,solveastro=solveastro,
            injectScale=injectScale,
            bigparallel=bigparallel,subprocess_count=subprocess_count,
            runoutlier=runoutlier,getmedianimg=getmedianimg,bigmedianimg=medianimg,minimg=minimg,p1=p1,p2=p2)
        tlist.append(time.time() - stime)
    print('TIME:')
    #print(tlist)
    return output1, output2,outputP1,outputP2

def associate_all(satfolder,satlist, framerange, t, median, binfactor,bigparallel,subprocess_count,injectScale,
    winlist,width,overlap,runoutlierpair=[0,1]):
    '''    Demo of GEO navstar observation, tighter angular tolerance for detection   '''
    #SPECIFY DATA INPUT AND FOLDERS
    #satfolder= 'data' #folder of all data
    #satlist = ['20201224_26407_navstar-48',] #folder of test data
    datatype='fits' #test data suffix
    #datatype='fit.zip' #test data suffix
    #imgfolder='results_NavstarDemo' #where to store results
    #framerange = [-1,18] #frames of test data to use, default all [-1,-1]
    imscale = 1 #scale of test data to import (crop precentage)
    numsteps = 5 #number of partitioning parallelization steps (numsteps-x-numsteps partitioning)
    #if binfactor==1:
    #    numsteps=9
 
    #PRINTING OPTIONS   
    printonly = 0 #flag to only print input data and exit (initial visualization)
    makeimg = 1 #flag to print still images
    makegif = 1 #flag to print animations
    printcluster=1 #flag to print 2nd order clustering
    vs = 0.25 #scale of printing (precentage)
    fps=5 #fps of animation .gif
    

    #PIPELINE FLOW OPTIONS
    runoutlier=runoutlierpair[1]
    solvemarkov=0 #flag solve local markov kernels or assume global uniformity
    resolve=1 #flag to enable/disable resolving of 1st-order line detections
    cluster = 1 #flag to disable(0)/ resolve (1)/ or use existing (2) 2nd-order clustering 
    solveastro=1#2 #flag to disable(0)/ resolve (1)/ or use existing (2) astrometry solution
    imgastro=3#0#2 #flag to use image data in ranking star values (1) or rank by NFA (0)
    tle=[] #optional TLE to use, if present generates an 'expected track' for precision-recall analysis

    #HYPERPARAMETERS
    a=0.0 #gradient kernel for 1st-order line detection.  3x3 sobel kernel (0) or radius of GR kernel.
    #t=1.3 #tightening factor for 1st-order angular tolerance (threshold tau/t)
    #median = 0 #flag to subtract the median value of each pixel, e.g. to remove hot pixels
    #binfactor=1 #integer binning factor 

    #OTHER PARAMETERS
    e2 = 0 #meaninfulness threshold of clustering algorithm, used in precision-    recall analysis

    #RUN PYCIS
    tlist = [] #record runtime for multiple hyperparameter options
    #for dummy_parameter in [0,]: #may iterate over several hyperparameter options 
    imgnamelist=[]
    linenamelist=[]
    rangetag = 'ASSOC%dT%dW%dO%d'%(winlist[0][0]+1,winlist[-1][1]+1,width,overlap)   
    #allname = 'a%dt%dm%db%d_%s'%(a,int(t*10),median,binfactor,rangetag)
    #allname = 'W%dO%dNewmedian_2dZ1_t%d_spline'%(width,overlap,int(t*10))#'a%dt%dm%db%d_%s'%(a,int(t*10),median,binfactor,rangetag)
    allname = '%s'%(rangetag)
    #alllinename = '%s_e%d_inj%d'%(allname,int(e2*100),injectScale)
    alllinename = '%s'%allname#'%s_e%d_inj%d'%(allname,int(e2*100),injectScale)

    #Build list of memory to call 
    for framerange in winlist:
        rangetag = 'A%dB%d'%(framerange[0]+1,framerange[1]+1)
        imgname = 'a%dt%dm%db%d_%s'%(a,int(t*10),median,binfactor,rangetag) #image name to save, listing hyperparameter options
        linename = '%s_e%d_inj%d'%(imgname,int(e2*100),injectScale) #For PR analysis, can fix 1st-order detections and redo clustering for e2 options
        imgnamelist.append(imgname)
        linenamelist.append(linename)

    stime = time.time() 
    end_pipe(satfolder,satlist,datatype,numsteps,
        allname,alllinename, imgnamelist, linenamelist,winlist,
        imgfolder,tdmfolder,vs*float(binfactor),makegif,printcluster,
        resolve,imscale, median,shift=0,sig=0,e2=0,makeimg=makeimg,solveastro=solveastro,
        linename='NULL',binfactor=binfactor,fps=fps,tle=[],imgastro=imgastro,cluster=cluster,background=0,
        runoutlier=runoutlier,subprocess_count=subprocess_count)

    tlist.append(time.time() - stime)
    print('TIME:')
    print(tlist)


if __name__=="__main__":
    '''   
    Run example demo scripts.  Be sure to download input ASTRIANet data according to README.
    '''
    
    parser = argparse.ArgumentParser(description='ASTRIANET PyCIS Telescope Image Pipeline')

    parser.add_argument("-i",type=str,help='Input data directory')
    parser.add_argument("-s",type=str,help='Input data obs. subdirectory')
    parser.add_argument("-o",type=str,help='Output data directory')
    parser.add_argument("-t",type=str,help='Output TDM directory')
    parser.add_argument("-w",type=int,help='Window width')
    
    args = parser.parse_args()
        
    satfolder = args.i #folder of all data
    sat       = args.s #folder of all data
    satlist   = [sat,] #folder of test data
    imgfolder = args.o #where to store results
    tdmfolder = args.t #where to store results


    width=30 #frame for window
    overlap=0 #overlapping between windows
    bigparallel = 1
    subprocess_count = 20#20
    injectScale = -1 #0 #-1
    

    for width in [args.w,]:#[30,20,10]:#[20,30]:
        #overlap = width-int(np.ceil(width/3.))
        alltime = time.time() 
        #winlist, t, median, binfactor, runoutlierpair = getsettings(satfolder,sat,width,overlap)
        winlist, t, median, binfactor, runoutlierpair = getsettings(satfolder,sat,width,overlap)
        width2 = int(np.ceil(width*2./3.))
        winlistB, _, _, _, _ = getsettings(satfolder,sat,width2,overlap)
        winlist.extend(winlistB)
        print('FULL WINLIST:',winlist)
        injectScale=-1
        #Witha 20-wide window in WINLIST, 
        if binfactor==1:
            bigparallel=0

        print('WINDOWS:')
        print(winlist)
        allwintime = time.time()
        #'''
        medianimg = []
        minimg=0
        p1 = None
        p2 = None
        for frameidx, framerange in enumerate(winlist):
            frametime = time.time()
            print('RUNNING FRAMERANGE %d/%d'%(frameidx+1,len(winlist)))
            tempframe,tempmin,tempp1,tempp2 = main(satfolder,satlist, framerange, t, median, binfactor, 
                bigparallel,subprocess_count,injectScale,runoutlierpair,frameidx,medianimg,minimg,p1,p2)
            if frameidx==0: #Get median img data from the full dataset, more accurate than window-based flat
                medianimg =np.copy(tempframe)
                minimg=np.copy(tempmin)
                p1=np.copy(tempp1)
                p2=np.copy(tempp2)
            #main(satfolder,satlist, framerange, t, median, binfactor, 
            #    bigparallel,subprocess_count,injectScale,runoutlierpair)
            frametime = time.time()-frametime
            print('WINDOW TIME : %.2f minutes'%(frametime/60.))
        #'''
        allwintime = time.time() - allwintime
        print('ALL WINDOW TIME : %.2f minutes'%(allwintime/60.))
        print('Windows complete.  Running final association... ')
        frametime = time.time() 
        print('RUNNING FRAME ASSOCIATION')
        associate_all(satfolder,satlist, [], t, median, binfactor, 
            bigparallel,subprocess_count,injectScale,
            winlist, width, overlap,runoutlierpair)
        frametime = time.time()-frametime
        print('ALL MERGE TIME : %.2f minutes'%(frametime/60.))

        
        alltime=time.time()-alltime
        print('ALL RUN TIME : %.2f minutes'%(alltime/60.))
