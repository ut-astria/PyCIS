'''
PyCIS - Python Computational Inference from Structure

pylib/main_utils.py: Helper functions for main_pipe.py, including track association and star inference algorithms.

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
Date of Modification: March 3, 2022

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
import sys
import time
import numpy as np
# Import other python functions 
from pylib.detect_outliers_agg import associate_lines
from pylib.print_detections import interp_frame_xy
from pylib.detect_utils import my_filter, remove_matches_block


class HideOutput(object):
    '''
    Used to suppress a-contrario track association using the class provided at
    https:// stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    A context manager that block stdout for its scope, usage:
    with HideOutput():
        os.system('ls -l')
    '''
    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)
    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)  


def associate_multiscale(numscales,goodlines,fullshape,outfolder,tempname,depth=0,window=0,filt='None'):
    '''
    Performs track association using a-contrario method, using sub-function from detect_outliers_agg.py.
    This aids in 
        1) merging lines from window-based and parallelized analysis, where single detections have been divided by implementation.
        2) cases where the gradient kernel may not be robust enough to draw the entire length of certain high-noise features  
        3) handling sensor-caused temporal aliasing, i.e. objects/stars with disjointed streaks on several different frames
    
    The core association method is from: 
        "An Unsupervised Point Alignment Detection Algorithm"
        by J. Lezama, G. Randall, J.M. Morel and R. Grompone von Gioi,
        Image Processing On Line, 2015. http://dx.doi.org/10.5201/ipol.2015.126
    TODO: 
        filtering with 'None' sometimes uses 'XY' default - explicitly state that this is necessary for temporal aliasing 
    Inputs:
        numscales - the number of partitions for the X/Y axes (Z axis is partitioned by window-analysis only, not considered here)
            NOTE: This defaults to 20 if filt='Z' or 1 if filt='XY', to ensure stronger association of low-relative motion or high-relative motion events
            NOTE: Scales with go from finest to coarsest (ie 3,2,1) to build up larger features
        goodlines - the input set of lines, should be parameterized [x1,y1,z1,x2,y2,z2,...], z1<=z2.  
        fullshape - the [X,Y,Z] dimensions of the entire FITS data cube 
        outfolder/tempname - the folder/file name to save working data for the alignment algorithm
        depth - the number of scales to consider, if 0 defaults to all scales (i.e., 3,2,1 if numscales=3)
            NOTE: This should usually be 2 or 0, so that association can handle window overlap effectively.
        window - the size of window analysis to set elevation filters for filt processing
        filt  - 'None' default. 'XY' and 'Z' used  to control special pipelines in the pre-clustering stage when line density is very high.  
            NOTE: behavior for 'None' uses some 'XY' filtering requirements since most problems arise due to temporal aliasing regardless of tracking mode 
    Outputs:
        goodlines_old - an updated verion of the input set with associated lines
        goodlines_trim - the set of lines which was not considered in association (if filt!='None'), used for plotting debug purposes only
    '''

    alltime=time.time()#-atime
    print('INPUT TO ASSOC_MULTISCALE: ',len(goodlines))
    goodlines_old = np.copy(goodlines)
    goodlines_tempstore=[]
    if np.isnan(depth) or depth==0:
        depth=numscales
        spreadcorr=1
        extrapflag=False
    else:
        spreadcorr=1
        extrapflag=False

    #If using pre-cluster filtering, define lines to use
    if filt=='XY' or filt=='Z':
        locline=np.copy(goodlines_old)        
        spreadfactor = 1.#max(1., min(fullshape[0],fullshape[1])/fullshape[2])
        locline[:,5]=locline[:,5]*spreadfactor
        locline[:,2]=locline[:,2]*spreadfactor
        k_len=np.linalg.norm(locline[:,:3]-locline[:,3:6],axis=1)
        el = np.arccos(np.abs(locline[:,5]-locline[:,2])/k_len)
        k_len_xy=np.linalg.norm(locline[:,:2]-locline[:,3:5],axis=1)
        el_pix = np.abs(locline[:,5]-locline[:,2])
        filtertol = 0.
        filtertol = np.median(el*180./np.pi)-2.5
        if window==0:
            window=fullshape[2]
        filtertol_pix = window*0.2 if filt=='Z' else 3 #np.median(el_pix)
       
        ktemp = k_len_xy[el_pix<=filtertol_pix] #get the 10%quantile
        filtertol_xy = sum(ktemp*ktemp)/sum(ktemp)
        filtertol_xy = np.median(ktemp) 
        #important because it averages by long lines are weighted much stronger, will trim a lot of noise 

        numvert = 0
        if 1==1:#while numvert<(.1*len(goodlines)):
            filtertol+=2.5            
            filteridx = (el*180./np.pi) < filtertol #low-velocity association 
            filteridx = (el_pix>filtertol_pix) if filt=='Z' else ((el_pix<=filtertol_pix) & (k_len_xy>filtertol_xy))
            print('considering %d/%d for %s  association at %.2f Z, %.2f XY pixels'%(np.count_nonzero(filteridx),len(filteridx),filt,filtertol_pix,filtertol_xy))
            numvert = np.count_nonzero(filteridx)
            
        #Save the unused lines to add back to set later
        goodlines_tempstore = goodlines_old[filteridx!=1,:]
        goodlines_old = goodlines_old[filteridx,:]
        #Force numscales for these pipelines 
        numscales = 20 if filt=='Z' else 1  #5percent width #int(np.floor(np.sqrt(len(goodlines_old)))), 20 percent for XY 
        #depth=numscales
        spreadcorr=1
        counter = len(goodlines_old)
        extrapflag=False

    #Iterate over the scales from finest to coarsest
    for depthidx, multiscale in enumerate(range(1,numscales+1)[::-1]):
        cc = fullshape[2]
        goodlines_extrap = np.copy(goodlines_old)
        #option to consider extrapolated lines in association, DO NOT USE
        if extrapflag:
            for k in range(len(goodlines_extrap)):
                locline = goodlines_extrap[k,:6].squeeze()
                x1,y1 = interp_frame_xy(locline,0,double=True,extrap=True)
                x2,y2 = interp_frame_xy(locline,cc,double=True,extrap=True)
                goodlines_extrap[k,:6] = np.copy([x1,y1,0,x2,y2,cc])
        atime=time.time()
        #Check if using or skipping depth 
        if (depthidx+1)>depth:
            print('skipping depth %d'%multiscale,flush=True)
            continue 
        else:
            print('depth = %d, OLD %d ...'%(multiscale,len(goodlines_extrap)),end=' ',flush=True)

        
        if multiscale==1:
            #Run association on full scale (unpartitioned)
            goodlines_all = np.copy(goodlines_extrap) #remove the 'main associations' as the stars should skew statistics to very long alingments only 
            try:
                with HideOutput():
                    _, goodlines, rejlines, _, align_diam = associate_lines(np.arange(len(goodlines_all)), goodlines_all,len(goodlines_all), fullshape,
                            postcluster=2, starfilter=0, densityfilter=0, folder=outfolder, name=tempname,spreadcorr=spreadcorr,
                            returndensity=True,newonly=True)
            except Exception as e:
                print(e,end=' ')
                goodlines=np.asarray([])
                rejlines=np.asarray([])
                align_diam=np.asarray([])
                continue 
            align_diam = np.asarray(align_diam).flatten()
        else:
            #Run association on some finer scale (partitions of window)
            numsteps = multiscale
            xstep = int(fullshape[0]/numsteps)
            ystep = int(fullshape[1]/numsteps)
            newshape=[xstep,ystep,fullshape[2]]
            goodlines_winset=[]
            align_diam = []
            iterable=[]
            xylist = []
            raisecnt=0
            allcnt=0
            #GO over each partition
            for xi,xidx in enumerate(range(0, fullshape[0], xstep)):
                for yi,yidx in enumerate(range(0,fullshape[1], ystep)):  
                    #print('starting %d/%d...'%(xi,yi),end=' ')
                    if ((xidx+xstep)>fullshape[0]) or ((yidx+ystep)>fullshape[1]):
                        continue
                    #Isolate the lines within this partition, note this will only consider lines fully within the partition 
                    allcnt+=1
                    goodlines_all = np.copy(goodlines_extrap)
                    goodlines_all[:,0]-=xidx; goodlines_all[:,3]-=xidx
                    goodlines_all[:,1]-=yidx; goodlines_all[:,4]-=yidx
                    goodlines_all = my_filter(goodlines_all,newshape,skipper=False,buffer=1.)
                    if len(goodlines_all)==0:
                        continue
                    #Ensure there are not any duplicate lines in this set, and run association
                    goodlines_all,_ = remove_matches_block(goodlines_all, goodlines_all,newshape[2],identical=True) #lines1=goodlines_old, lines2=goodlines
                    try:
                        with HideOutput():
                            _, goodlines_win, rejlines_win, _, align_diam_out = associate_lines(np.arange(len(goodlines_all)), goodlines_all,len(goodlines_all), newshape,
                                                    postcluster=2, starfilter=0, densityfilter=0, folder=outfolder, name=tempname,spreadcorr=spreadcorr,
                                                    returndensity=True,newonly=True)
                    except Exception as e:
                        raisecnt+=1
                        print(e,end=' ')
                        continue
                    try:     
                        #enforce behavior 
                        if not len(goodlines_win)==len(align_diam_out):
                            #this means goodlines is a backup, there are no associations!
                            continue
                        if len(goodlines_win)>0:
                            align_diam_temp = align_diam_out
                            if len(goodlines_tempstore)>0:
                                #make sure line obeys the filtering restrictions
                                locline = np.copy(goodlines_win)
                                locline[:,5]=locline[:,5]*spreadfactor
                                locline[:,2]=locline[:,2]*spreadfactor
                                k_len=np.linalg.norm(locline[:,:3]-locline[:,3:6],axis=1)
                                el = np.arccos(np.abs(locline[:,5]-locline[:,2])/k_len)
                                k_len_xy=np.linalg.norm(locline[:,:2]-locline[:,3:5],axis=1)
                                el_pix = np.abs(locline[:,5]-locline[:,2])
                                filteridx = (el_pix>filtertol_pix) if filt=='Z' else ((el_pix<=filtertol_pix) & (k_len_xy>filtertol_xy))
                                goodlines_win = goodlines_win[filteridx,:]
                                align_diam_temp = align_diam_temp[filteridx]    
                            #restore lines from the partition to the full set
                            goodlines_win[:,0]+=xidx; goodlines_win[:,3]+=xidx
                            goodlines_win[:,1]+=yidx; goodlines_win[:,4]+=yidx
                            goodlines_winset.append(np.copy(goodlines_win))
                            align_diam.append(np.asarray(align_diam_temp).flatten())
                    except Exception as e:
                        pass
            #Merge the partition sets
            atime=time.time()-atime
            if len(goodlines_winset)>0:
                goodlines = np.vstack(goodlines_winset)
                align_diam = np.concatenate(align_diam)
            else:
                goodlines=np.asarray([])
                rejlines=np.asarray([])
                align_diam=np.asarray([])
                continue 

        #REMOVE_MATCHES
        if (len(goodlines_old)==0) or (len(goodlines)==0):
            continue 
        goodlines_old,_ = remove_matches_block(goodlines_old,goodlines,window,diam=align_diam)

        #Make sure there aren't any duplicates at this step - implementation check 
        print('NEW %d ... '%(len(goodlines_old)),end=' ')
        goodlines_old,_ = remove_matches_block(goodlines_old, goodlines_old,fullshape[2],identical=True,septol=0.05*min(fullshape[0],fullshape[1])) #lines1=goodlines_old, lines2=goodlines
        print('RE-FILTERED %d '%(len(goodlines_old)))
        atime=time.time()#-atime

    #Add in the unconsidered lines 
    if len(goodlines_tempstore)>0:
        goodlines_trim = np.copy(goodlines_old)
        goodlines_old = np.vstack([goodlines_trim, goodlines_tempstore]) #merge the old unmatched and the new associated lines 
    else:
        goodlines_trim=[]
    print('OUTPUT FROM ASSOC_MULTISCALE: ',len(goodlines_old),flush=True)
    alltime=time.time()-alltime
    print('time %.2f min'%(alltime/60))
    return goodlines_old,goodlines_trim

def apriori_filter_star_frames_adaptwide(lines,zmax,septol,med_az=None,getstars=False):
    '''
    Identify and remove lines likely corresponding to well-detected star features, to 
        1) reduce computational load in clustering step 
        2) identify star properties to mitigate associated star feature detections after outlier detection 
        3) determine the best candidates for astrometric calibration in run_astrometry.py 
    This method is "adaptive" in that, for each frame, it explores the window of frames over which local orientation can be best established to remove the most lines.

    Input: 
        lines: the set of line features
        zmax: the number of frames in the data set
        septol: Used in removing matches in an effective runtime 
        med_az: Input for the previously found star orientations to remove associated lines after clustering/outlier
        getstars: Flag to return star features of first-order detections for run_astrometry.py 
    Output: 
        goodlines_out: The line set with star features removed
        med_az_out: The orientation of stars at each frame for later filtering
        [stars_out]: If getstars=True, the star features to be handed to run_astrometry.py 
    '''
    initlen = len(lines)
    medflag=True
    counter=0
    span=zmax*.1
    check = 0
    stacklen=0
    #Array used to identify the star set using indexing for effective speed 
    allstars = np.zeros(initlen).astype(dtype=bool) #start assuming no stars 
    lenopt=[]
    goodlocopt=[]
    medazopt=[]

    #Set the windows to explore when estimating optimal width for orientation estimation 
    fslist = [0,1]#1,3,span,zmax]
    fsmin=0
    if med_az is None:
        fsmincount=np.copy(initlen)
        fslist = np.arange(0,np.ceil(zmax*0.1)).astype(dtype=int).tolist()
        fslist.append(zmax)
    else: #we have the prior angles so don't need to bother with any averaging
        fslist=[0,] 
    print('Frame=window:',end=' ')
    med_az_set=[]
    
    #Iterate over each frame
    for i,frame in enumerate(range(zmax)):
        locstaridxlist = []
        medlist = []
        starcnt=0
        starcntidx=-1
        #Iterate over each window option for this frame
        for fsidx,fs in enumerate(fslist): #for fs in enumerate(fslist):#int(np.floor(zmax/2))):
            #Get lines for this window
            frame=float(frame)
            fa = np.amin(lines[:,[2,5]],axis=1); fb = np.amax(lines[:,[2,5]],axis=1)
            idx = ((fa-fs)<=frame) & (frame<=(fb+fs))
            loc = np.copy(lines[idx,:]) #may result in many overlaps 
            #Get the indices identifying stars 
            locstaridx = np.zeros(initlen).astype(dtype=bool)
            if med_az is None:
                goodloc,med_az_loc, staridx = apriori_filter_star(loc,med_az=None) 
            else:
                goodloc,med_az_loc, staridx = apriori_filter_star(loc,med_az=med_az[i]) 
            #From the window, isolate only those stars corresponding to this frame 
            fa = np.amin(goodloc[:,[2,5]],axis=1); fb = np.amax(goodloc[:,[2,5]],axis=1)
            idx2b = ((fa-0)<=frame) & (frame<=(fb+0))
            #Identify star indices and record results 
            staridx[~idx2b] = False #cancel those lines outside the range
            locstaridx[idx]=staridx #after getting stars, reduce to those truly intersecting frame of interest
            locstaridxlist.append(locstaridx)
            medlist.append(med_az_loc)
            if np.count_nonzero(locstaridx)>starcnt:
                starcnt=np.copy(np.count_nonzero(locstaridx))
                starcntidx=np.copy(fsidx)

        #Update star lists with optimal window setting 
        if med_az is None:
            print('%d=%d,'%(frame,fslist[starcntidx]),end=' ',flush=True )
        locstaridx = locstaridxlist[starcntidx]
        allstars = allstars | locstaridx
        stacklen+=len(goodloc)
        med_az_set.append(medlist[starcntidx])
    print(' ')
    goodlines_out = np.copy(lines)
    if getstars:
        stars_out = goodlines_out[allstars,:]
    goodlines_out = goodlines_out[~allstars,:]
    med_az_out = np.vstack(med_az_set) #merge the old unmatched and the new associated lines 
    stacklen2=len(goodlines_out)
    timeA = time.time()

    #Be sure there are no duplicated lines on the rejected set, now that the population should be reduced enough for effective runtime
    #If runtime is not effective, there are too many first-order detections to solve, and LSD setttings should be revisited
    endlenA = np.copy(len(goodlines_out))
    goodlines_out,_ = remove_matches_block(goodlines_out, goodlines_out,zmax,identical=True,septol=septol,runspliner=True,fullcheck=False) #lines1=goodlines_old, lines2=goodlines
    timeA = time.time()-timeA
    print('FULLCHECK=FALSE TIME: %.2f'%(timeA/60.),flush=True)
    if timeA/60. > 5.:
        print('ERROR: First-layer filtering in apriori_filter_star_frames_adaptwide() takes too long.')
        print('       The LSD algorithm yeilds too many results at current setting, and will fail in clustering.')
        print('       Aborting for now for safety.')
        quit()
    #Stars will also need filtered before feeding them to run_astrometry.py 
    if getstars:
        print('STAR REDUCTION %d ->'%(len(stars_out)),end=' ')
        stars_out,_ = remove_matches_block(stars_out, goodlines_out,zmax,identical=True,septol=septol,runspliner=True,fullcheck=False) #lines1=goodlines_old, lines2=goodlines
        print('%d'%(len(stars_out)))
    #We now run a second search for duplicates which can quickly merge obvious connected lines (same orientation and touching) to save runtime on association
    endlenB = np.copy(len(goodlines_out))
    print('REDUCTION %d -> %d'%(endlenA,endlenB))
    timeA=time.time()
    print('SIMPLE REPALCEMENT ONLY, ALLOW SPLINER BUT ONLY CONSIDER ENDPOINT 1')
    goodlines_out,_ = remove_matches_block(goodlines_out, goodlines_out,zmax,identical=True,septol=septol,runspliner=True,fullcheck=True) #lines1=goodlines_old, lines2=goodlines
    timeA = time.time()-timeA
    print('FULLCHECK=True TIME: %.2f'%(timeA/60.),flush=True)
    outlen = len(goodlines_out)
    print('REDUCTION %d -> %d -> %d'%(endlenA,endlenB,outlen))
    print('COMPLEX STAR FILTER: init %d, stack %d,filter %d, output/match %d'%(initlen,stacklen,stacklen2,outlen))
    #Output
    if getstars:
        return goodlines_out, med_az_out, stars_out
    return goodlines_out, med_az_out

def apriori_filter_star(lines,med_az=None):
    '''
    Helper function for star filtering.
    Identify star-like features, matching a length-weighted mean azimuth.
    TODO: This is designed assuming rate-tracking, and will need corrected for sidereal-tracking!!!
    '''
    #get statistics 
    k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
    k_len_xy= np.linalg.norm(lines[:,:2]-lines[:,3:5],axis=1)
    el = np.arccos((lines[:,5]-lines[:,2])/k_len)
    el_pix = np.abs(lines[:,5]-lines[:,2])
    az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
    #OBSOLTE: Would check for near-horizontal, rate-tracking only. 
    med_el=3
    idxE = el_pix <= med_el #much more restrictive 
    idxE[:]=True #if we know star prior, only consider azimuth, usually after clustering 
    #identify length-weighted mean azimuth  (only need xy direction, roughly corresponding to temporal aliasing)
    aztemp = az[idxE] #median azimuth is the star angle, where elevation valid
    ktemp = k_len_xy[idxE]**2 #really crack this up to enforce obeying, there could be a ton of noise
    if len(ktemp)==0:
        #Abort if there are no lines in this set
        return lines, 4.*np.pi, np.zeros(len(lines)).astype(dtype=bool)
    if med_az is None:
        med_az = sum(aztemp*ktemp)/sum(ktemp) #average az weighted by xy-length 
    
    #Identify set "aligned " with orientation, using LSD definition.
    tol = 22.5
    idxA = np.abs(az-med_az) < (tol*np.pi/180.)
    if not (med_az is None): #antiparallel catch 
        idxA2 = np.abs((az-np.pi)-med_az) < (tol*np.pi/180.)
        idxA3 = np.abs((az+np.pi)-med_az) < (tol*np.pi/180.)
        idxA = (idxA | idxA2) | idxA3
    #Return star lines, the azimuth used, and the index array used for runtime
    idx = idxA & idxE
    idx=idx.astype(dtype=bool)
    return lines, med_az, idx


def apriori_filter(lines):
    '''
    A helper function for post-clustering/pre-outlier.  Remove known noise features:
        1) hot/dead pixels.  These have zero image-frame motion, which is unrealistic in any tracking motion 
        2) Feautures with less than 3 frames of measurement, cannot be used for Initial Orbit Determination. 
    '''
    #remove pure-vertical lines, apriori difference b/w hot pixels and object tracking noise
    idxA1 = np.abs(lines[:,0]-lines[:,3])<1
    idxA2 = np.abs(lines[:,1]-lines[:,4])<1
    idxA = idxA1 & idxA2
    #remove associated lines which do not have IOD capability (less than 3 time observations)
    idxB = np.abs(lines[:,2]-lines[:,5])<3
    #apply filter
    idx = idxA | idxB
    print('num hot pixels: ',np.count_nonzero(idxA))
    print('num invalid IOD: ',np.count_nonzero(idxB))
    print('apply filter, will remove %d / %d '%(np.count_nonzero(idx), len(lines)))
    lines = lines[idx!=1,:]
    return lines
