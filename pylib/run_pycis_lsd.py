'''
PyCIS - Python Computational Inference from Structure

pylib/run_pycis_lsd.py: Pipeline for detecting linear feautures by calling the LSD functions of PyCIS

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
Date of Modification: February 16, 2022

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
import os
import time
import numpy as np


#blockprocessor
from multiprocessing import Pool, cpu_count,get_context,set_start_method
#Call the LSD function
from pylsd.pycis_lsd_wrapper import main as pycis_lsd_wrapper
 


def run_pycis_lsd(
    I3, outfolder, prename,
    numsteps,bigparallel,
    solvemarkov,resolve,printonly,
    a,tc):
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
    #FEED:
    #I3 after contiguous
    #Name
    #outfolder
    #a
    #t (tc)

    #LSD settings - Gradeint-by-ratio and centerline angular tolerance are controlled
    #a = a  #Gradient-by-Ratio parameter 
    ae = a; ac=a  #Gradient-by-Ratio parameter 
    t = 1.0  #Division factor for angular tolerance
    te=t; #tc=t


    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################


    ## Things that are contained within PyCIS-LSD

    #ALL DENSITY IS ZERO
    d = 0.0 #Density threshold 
    de = d; dc = d;#0.05
    #LSD DEFAULTS TO ZERO
    e=0;
    #at a=0.5, good at de=0.6, dc=0.0
    #DEFAULT TO NO GAUSSIAN
    scale=1. #Gaussian downsampling volume ratio 
    sigma=.6 #Gaussian downsampling deviation factor

    #Image and parallelization parameters
    shaper = (I3.shape[0], I3.shape[1], I3.shape[2])
    emptyset = np.eye(2)
    memsize = I3.size*I3.itemsize/1.0e9
    print('NUMSIZE:',I3.shape)
    print('MEMSIZE:',memsize,'GB')
    #Fix name

    xstep = int(I3.shape[0]/numsteps)
    ystep = int(I3.shape[1]/numsteps)

    if bigparallel==0:  
        process_count = numsteps
        process_count = min(min(process_count,numsteps**1),cpu_count())
        chunks = min(1,int(numsteps/process_count))
    else:
        process_count = numsteps**2
        process_count = min(min(process_count,numsteps**2),cpu_count())
        chunks = min(1,int((numsteps**2)/process_count))


    print('cpus: %d / %d'%(process_count,cpu_count()))
    p=[0,0,0,0,0,0,0,0,0,0,0,0]
    p2=[0,0,0,0,0,0,0,0,0,0,0,0]


    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################

    ## COMPUTE MARKOV KERNELS 

    if solvemarkov==1:
        ## COMPUTE FOR EACH PARTITION 
        #Edge line kerneel
        pset=np.zeros((numsteps,numsteps,12))
        iterable=[]
        a=ae;t=te;
        for xi,xidx in enumerate(range(0, I3.shape[0], xstep)):
            if bigparallel==0:
                iterable=[]
            for yi,yidx in enumerate(range(0,I3.shape[1], ystep)):  
                name = "%d_%d_%s"%(xi,yi,prename)
                if ((xidx+xstep)>I3.shape[0]) or ((yidx+ystep)>I3.shape[1]):
                    continue
                if os.path.exists('%s/data1_%s.npy'%(outfolder,name)) and resolve==0:
                    continue 
                I3b = I3[xidx:xidx+xstep, yidx:yidx+ystep, :]
                iterable.append((np.copy(emptyset), np.copy(I3b),outfolder,name,
                    a,d,t,scale,sigma,e,2,p,p2,(xi,yi),shaper))
            if bigparallel==0:
                with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                    results=pool.starmap(pycis_lsd_wrapper,iterable,chunksize=chunks)
                for r in results:
                    idxs = np.array(r[0])
                    print(idxs)
                    vals = np.array(r[1])
                    print(vals)
                    outv = np.asarray(vals[10:])
                    pset[int(idxs[0]),int(idxs[1]),:] = outv[np.newaxis,np.newaxis,:]
        if not bigparallel==0:
            with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                results=pool.starmap(pycis_lsd_wrapper,iterable,chunksize=chunks)
            for r in results:
                idxs = np.array(r[0])
                print(idxs)
                vals = np.array(r[1])
                print(vals)
                outv = np.asarray(vals[10:])
                pset[int(idxs[0]),int(idxs[1]),:] = outv[np.newaxis,np.newaxis,:]

        #Center line kernel
        p2set=np.zeros((numsteps,numsteps,12))
        iterable=[]
        a=ac;t=tc;
        for xi,xidx in enumerate(range(0, I3.shape[0], xstep)):
            if bigparallel==0:
                iterable=[]
            for yi,yidx in enumerate(range(0,I3.shape[1], ystep)):  
                name = "%d_%d_%s"%(xi,yi,prename)
                if ((xidx+xstep)>I3.shape[0]) or ((yidx+ystep)>I3.shape[1]):
                    continue
                if os.path.exists('%s/data1_%s.npy'%(outfolder,name)) and resolve==0:
                    continue 
                I3b = I3[xidx:xidx+xstep, yidx:yidx+ystep, :]
                iterable.append((np.copy(emptyset), np.copy(I3b),outfolder,name,
                    a,d,t,scale,sigma,e,3,p,p2,(xi,yi),shaper))
            if bigparallel==0:
                with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                    results=pool.starmap(pycis_lsd_wrapper,iterable,chunksize=chunks)
                for r in results:
                    idxs = np.array(r[0])
                    print(idxs)
                    vals = np.array(r[1])
                    print(vals)
                    outv = np.asarray(vals[10:])
                    p2set[int(idxs[0]),int(idxs[1]),:] = outv[np.newaxis,np.newaxis,:]
        if not bigparallel==0:
            with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                results=pool.starmap(pycis_lsd_wrapper,iterable,chunksize=chunks)
            for r in results:
                idxs = np.array(r[0])
                print(idxs)
                vals = np.array(r[1])
                print(vals)
                outv = np.asarray(vals[10:])
                p2set[int(idxs[0]),int(idxs[1]),:] = outv[np.newaxis,np.newaxis,:]
                
    elif not (os.path.exists('%s/data1_%s.npy'%(outfolder,prename)) and resolve==0):
        name=prename
        ## COMPUTE FOR ONE PARTITION AND ASSUME UNIFORM
        pset=np.zeros((numsteps,numsteps,12))
        p2set=np.zeros((numsteps,numsteps,12))
        I3b = I3[:xstep, :ystep, :]
        # Edge line kernel
        a=ae;d=de;t=te;
        iterable=[(np.copy(emptyset), np.copy(I3b),outfolder,name,
                    a,d,t,scale,sigma,e,2,p,p2,(0,0),shaper)]
        with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                results=pool.starmap(pycis_lsd_wrapper,iterable,chunksize=1)
        for r in results:
            outv = np.array(r[1])
        p = np.copy(np.asarray(outv[10:]))
        #Center line kernel
        a=ac;d=dc;t=tc;
        iterable=[(np.copy(emptyset), np.copy(I3b),outfolder,name,
                    a,d,t,scale,sigma,e,3,p,p2,(0,0),shaper)]
        with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                results=pool.starmap(pycis_lsd_wrapper,iterable,chunksize=1)
        for r in results:
            outv = np.array(r[1])
        p2 = np.copy(np.asarray(outv[10:]))
        #Merge
        for xi,xidx in enumerate(range(0, I3.shape[0], xstep)):
            for yi,yidx in enumerate(range(0,I3.shape[1], ystep)):  
                if ((xidx+xstep)>I3.shape[0]) or ((yidx+ystep)>I3.shape[1]):
                    continue
                pseti = np.copy(np.asarray(p))
                pset[xi,yi,:] = pseti[np.newaxis,np.newaxis,:] 
                p2seti = np.copy(np.asarray(p2))
                p2set[xi,yi,:] = p2seti[np.newaxis,np.newaxis,:]
    

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################


    ## COMPUTE LINE FEATURES
    prenamecopy = prename        
    p[:6]=p2[:6]
    
    ## COMPUTE EDGE LINES
    a=ae;t=te;
    d=de
    iterable=[]
    for xi,xidx in enumerate(range(0, I3.shape[0], xstep)):
        if bigparallel==0:
            iterable=[]
        for yi,yidx in enumerate(range(0,I3.shape[1], ystep)):  
            name = "%d_%d_%s"%(xi,yi,prename)
            if ((xidx+xstep)>I3.shape[0]) or ((yidx+ystep)>I3.shape[1]):
                continue
            if os.path.exists('%s/data1_%s.npy'%(outfolder,name)) and resolve==0:
                continue 
            I3b = I3[xidx:xidx+xstep, yidx:yidx+ystep, :]
            p = pset[xi,yi,:].squeeze()
            p2 = p2set[xi,yi,:].squeeze()
            iterable.append((np.copy(I3b), np.copy(emptyset),outfolder,name,
                a,d,t,scale,sigma,e,0,p,p2,(xi,yi),shaper))
        if bigparallel==0:
            with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                pool.starmap(pycis_lsd_wrapper,iterable,chunksize=chunks)          
    if not bigparallel==0:
        with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
            pool.starmap(pycis_lsd_wrapper,iterable,chunksize=chunks)          

    
    ## COMPUTE CENTER LINES (use edge line priors)
    d=dc
    a=ac;t=tc;
    iterable=[]
    for xi,xidx in enumerate(range(0, I3.shape[0], xstep)):
        if bigparallel==0:
            iterable=[]
        for yi,yidx in enumerate(range(0,I3.shape[1], ystep)):  
            name = "%d_%d_%s"%(xi,yi,prename)
            if ((xidx+xstep)>I3.shape[0]) or ((yidx+ystep)>I3.shape[1]):
                continue
            if os.path.exists('%s/data2_%s.npy'%(outfolder,name)) and resolve==0:
                continue 
            I3b = I3[xidx:xidx+xstep, yidx:yidx+ystep, :]
            lines = np.load("%s/data1_%s.npy"%(outfolder,name))
            if len(lines)<1:#00:
                print('FEWER THAN 1 LINES')
                continue
            p = pset[xi,yi,:].squeeze()
            p2 = p2set[xi,yi,:].squeeze()
            iterable.append((np.copy(I3b), np.copy(lines),outfolder,name,
                a,d,t,scale,sigma,e,0,p,p2,(xi,yi),shaper))
        if bigparallel==0:
            with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                pool.starmap(pycis_lsd_wrapper,iterable,chunksize=chunks)
    if not bigparallel==0:
        with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
            pool.starmap(pycis_lsd_wrapper,iterable,chunksize=chunks)


    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################


    l2 = np.array([]).reshape(0,10)
    ## MERGE EDEGE LINE FILES (from parallelization)
    if 1==1: #not os.path.exists('%s/data2_%s.npy'%(outfolder,prename)):
        for xi,xidx in enumerate(range(0, I3.shape[0], xstep)):
            for yi,yidx in enumerate(range(0,I3.shape[1], ystep)):  
                if ((xidx+xstep)>I3.shape[0]) or ((yidx+ystep)>I3.shape[1]):
                    continue
                if not os.path.exists("%s/data1_%d_%d_%s.npy"%(outfolder,xi,yi,prename)): 
                    continue
                l1 = np.load("%s/data1_%d_%d_%s.npy"%(outfolder,xi,yi,prename))
                l1 = np.reshape(np.asarray(l1).T, (-1,10), order='F')
                #print(len(l1))
                l1[:,0] = l1[:,0]+xidx
                l1[:,1] = l1[:,1]+yidx
                l1[:,3] = l1[:,3]+xidx
                l1[:,4] = l1[:,4]+yidx
                l2 = np.vstack([l2,l1])
        lines = np.reshape(np.asarray(l2), (-1), order='F')
        savename = "%s/data1_%s.npy"%(outfolder,prename)
        np.save(savename,lines)
    

    ## MERGE CENTER LINE FILES (from parallelization)
    l2 = np.array([]).reshape(0,10)
    if 1==1: #not os.path.exists('%s/data2_%s.npy'%(outfolder,prename)):
        for xi,xidx in enumerate(range(0, I3.shape[0], xstep)):
            for yi,yidx in enumerate(range(0,I3.shape[1], ystep)):  
                if ((xidx+xstep)>I3.shape[0]) or ((yidx+ystep)>I3.shape[1]):
                    continue
                if not os.path.exists("%s/data2_%d_%d_%s.npy"%(outfolder,xi,yi,prename)): 
                    continue
                l1 = np.load("%s/data2_%d_%d_%s.npy"%(outfolder,xi,yi,prename))
                l1 = np.reshape(np.asarray(l1).T, (-1,10), order='F')
                #np.set_printoptions(suppress=True)
                l1[:,0] = l1[:,0]+xidx #+0
                l1[:,1] = l1[:,1]+yidx #+0
                l1[:,3] = l1[:,3]+xidx #+0
                l1[:,4] = l1[:,4]+yidx #+0
                l2 = np.vstack([l2,l1])
        #print(l2[:5,:])
        lines=l2
        lines = np.reshape(np.asarray(l2), (-1), order='F')
        #print(lines[:50])
        savename = "%s/data2_%s.npy"%(outfolder,prename)
        np.save(savename,lines)
    else:
        if os.path.exists("%s/data2_%s.npy"%(outfolder,prename)): 
            lines = np.load("%s/data2_%s.npy"%(outfolder,prename))
        else:
            lines = np.load("%s/data1_%s.npy"%(outfolder,prename))


    return lines