'''
PyCIS - Python Computational Inference from Structure

pylib/detect_utils.py: Helper for main_pipe and detect_outliers_agg, includes removal of 'duplicate' lines


Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
Date of Modification: March 03, 2022

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
import time
from multiprocessing import cpu_count,get_context #, Pool
import numpy as np
np.set_printoptions(precision=2)
np.seterr(all='ignore')
import scipy as sp
from pylib.print_detections import interp_frame_xy
#import tempfile


def my_filter(lines,shape,skipper=True,buffer=0):
    '''Filter lines to make sure they are within [xyz] defined by shape, with a Buffer to handle edge cases'''
    if skipper:
        return lines
    idx1 = lines[:,0]>=buffer
    if not np.any(idx1):
        return []
    lines=lines[idx1,:]
    idx2 = lines[:,1]>=buffer
    if not np.any(idx2):
        return []
    lines=lines[idx2,:]
    idx3 = lines[:,2]>=buffer
    if not np.any(idx3):
        return []
    lines=lines[idx3,:]
    idx4 = lines[:,0]<=(shape[0]-buffer)
    if not np.any(idx4):
        return []
    lines=lines[idx4,:]
    idx5 = lines[:,1]<=(shape[1]-buffer)
    if not np.any(idx5):
        return []
    lines=lines[idx5,:]
    idx6 = lines[:,2]<=(shape[2]-buffer)
    if not np.any(idx6):
        return []
    lines=lines[idx6,:]
    idx1 = lines[:,3]>=buffer
    if not np.any(idx1):
        return []
    lines=lines[idx1,:]
    idx2 = lines[:,4]>=buffer
    if not np.any(idx2):
        return []
    lines=lines[idx2,:]
    idx3 = lines[:,5]>=buffer
    if not np.any(idx3):
        return []
    lines=lines[idx3,:]
    idx4 = lines[:,3]<=(shape[0]-buffer)
    if not np.any(idx4):
        return []
    lines=lines[idx4,:]
    idx5 = lines[:,4]<=(shape[1]-buffer)
    if not np.any(idx5):
        return []
    lines=lines[idx5,:]
    idx6 = lines[:,5]<=(shape[2]-buffer)
    if not np.any(idx6):
        return []
    lines=lines[idx6,:]
    return lines
    
def fixlines(lines):
    ''' Helper function to make sure z1<=z2 for quicker match checking '''
    linesIn=np.copy(lines)
    linesIdx = linesIn[:,2]>linesIn[:,5]
    lines[linesIdx,0:3] = np.copy(linesIn[linesIdx,3:6])
    lines[linesIdx,3:6] = np.copy(linesIn[linesIdx,0:3])
    del linesIn
    return lines
 


def spliner_sub(lines1,lines2,distMat,printdetail=False,azfilter=True):
    '''
    Helper function to spliner (merging slightly non-linear tracks), which 
    looks for admissible matching lines (similar az, perhaps dissimilar elevation), and 
    merges these lines with a bounding set of endpoints.  
    The resulting trajectory is less precise but properly associated, and may be passed to run_astrometry for post-processing.
    TODO: This step should not be necessary after adding non-linear a-contrario track association.

    Input:
        lines1/lines2: The two sets of lines to consider merging 
        distMat: the matrix informing which line1/line2 pairs should be checked, having a valid distance metric 
        printdetail: a flag to print some detail for debug
        azfilter: whether to only check distance when the azimuth are aligned 
    OUtput:
        lines1: the new line set, where among matching lines one is given a set of bounding endpoints
        whilecnt: the number of lines considered in the filter
        whilemarker: whether any lines were flagged for merging 
    '''
    #Instantiate flags for checking if method has ended sucessfully to pass to spliner
    whilemarker=False
    whilecnt=0
    #Vectorized access of lines which have some match and computation of their lengths/orientations
    distCheck = np.any(distMat,axis=1)
    kxy_1 = np.linalg.norm(lines1[distCheck,:2]-lines1[distCheck,3:5],axis=1)
    az_1 =  np.arctan2((lines1[distCheck,3]-lines1[distCheck,0]) , (lines1[distCheck,4]-lines1[distCheck,1]))
    distCheckIdx = np.argwhere(distCheck)
    distCheckIdx = distCheckIdx.flatten().tolist()
    az_2 =  np.arctan2((lines2[:,3]-lines2[:,0]) , (lines2[:,4]-lines2[:,1]))

    #Iterate over those lines which have at least some match.
    for ii,i in enumerate(distCheckIdx): #range(distMat.shape[0]): #iterate over lines1 and merge with any matches in lines2 (not other lines1)
        whilecnt+=1
        #select those lines which register as a "match"
        kxy_i = kxy_1[ii] #i/j and az1/az2 when switching lead 
        az_i = az_1[ii] #i/j and az1/az2 when switching lead 
        #el_i = el_1[i]
        #Identify the candidate matches which have sufficient orientation (if using azfilter)
        distVec = distMat[i,:] #preserved with distMat.T when switching lead 
        azVec = np.abs(az_i - az_2) < (22.5*np.pi/180.) #always checks against the follower
        #elVec = np.abs(el_i - el_2) < (22.5*np.pi/180.)
        if azfilter:
            matchVec = distVec & azVec # (azVec & elVec)
        else:
            matchVec = distVec# & azVec # (azVec & elVec)
        
        #Among valid candidates, choose a set of bounding endpoints and check that the new line is both longer and properly aligned
        if np.count_nonzero(matchVec)>0:
            lines2_match = lines2[matchVec,:] #always match with the follower
            kxy_2 = np.linalg.norm(lines2[matchVec,:2]-lines2[matchVec,3:5],axis=1)
            kxy_match = max(kxy_2)#[matchVec]) #dont want to overwrite if the next lead/follow switch will be better! 

            oldline=np.copy(lines1[i,:])
            lines2_match = np.vstack([lines2_match,lines1[i,:]]) 
            #print('z2options:',lines2_match)
            zmin = np.argmin(lines2_match[:,2])
            zmax = np.argmax(lines2_match[:,5])
            lines1_new = np.copy(lines1[i,:])
            lines1_new[:3] = np.copy(lines2_match[zmin,:3])
            lines1_new[3:6] = np.copy(lines2_match[zmax,3:6])
            lines1_new[6:] = np.copy(np.amax(lines2_match[:,6:],axis=0))        
            #print('lines1new:',lines1_new)

            kxy_new = np.linalg.norm(lines1_new[:2] - lines1_new[3:5]) 
            az_new = np.arctan2((lines1_new[3]-lines1_new[0]) , (lines1_new[4]-lines1_new[1]))
            if azfilter:
                azpass=(np.abs(az_new - az_i) < (22.5*np.pi/180.))
            else:
                azpass=True

            if azpass and (kxy_new > max(kxy_i,kxy_match)):
                lines1[i,:] = np.copy(lines1_new)
                #distMat won't be updated now, replacement will simply utilize the K switch we previously wrote
                #swaping s.t. the longest line among all matches is preserved 
                if np.any(lines1[i,:]!=oldline): #rerun once more if an update has been made this iteration 
                    whilemarker=True
            else:
                pass#print('new,i,match:',[kxy_new,kxy_i,kxy_match])
    return lines1 ,whilemarker,whilecnt 


def spliner(lines1,lines2,distMat,printdetail=False,azfilter=True):
    ''''
    Looks for admissible matching lines (similar az, perhaps dissimilar elevation), and 
    merges these lines with a bounding set of endpoints.  
    The resulting trajectory is less precise but properly associated, and may be passed to run_astrometry for post-processing.
    TODO: This step should not be necessary after adding non-linear a-contrario track association.
    
    Input:
        lines1/lines2: The two sets of lines to consider merging 
        distMat: the matrix informing which line1/line2 pairs should be checked, having a valid distance metric 
        printdetail: a flag to print some detail for debug
        azfilter: whether to only check distance when the azimuth are aligned 
    OUtput:
        lines1/lines2: the new line set, where among matching lines one is given a set of bounding endpoints
        distMat: returns the same matrix, as the longest line among matches will be returned 
    '''
    #Prepare lines making sure endpoints are ordered z1<=z2 for each set
    try:
        lines1=fixlines(lines1)    
        lines2=fixlines(lines2)    
    except Exception as e:
        print('error calling fixlines within spliner')
        print(e)
    whilemarker=True
    #until we don't improve:
    #TODO: while loop must set to false if we NEVER go into a count_nonzero if statement.  
    cnt1all=0;cnt2all=0;
    while whilemarker:
        if printdetail:
            print('lines1:',lines1)
            print('lines2:',lines2)
            print('doing 1...')
        #Until there is no change, switch between checking if lines1 can be merged with lines2 and vice versa 
        lines1,whilemarker1,cnt1 = spliner_sub(lines1,lines2,distMat,printdetail,azfilter)
        if printdetail:
            print('doing 2...')
        lines2,whilemarker2,cnt2 = spliner_sub(lines2,lines1,distMat.T,printdetail,azfilter)
        cnt1all+=cnt1; cnt2all+=cnt2;
        whilemarker=whilemarker1|whilemarker2 #if either one was changed will need to repeat 
    return lines1,lines2,distMat

   
def distmat_sub(i,j,cc,goodlines_i,goodlines_j,align_diam_j,printdetail=False,fullcheck=True,azfilter=True,extracheck=False):
    '''
    Compute the distance metric between two lines.
    Inputs:
        i,j: The indices of the total line set from which the two lines are drawn
        cc: The number of frames in the data set the lines exist in
        goodlines_i/j: the two lines being compared
        align_diam_j: the diameter of the line feature j used in determining if the distance is valid and should have a "passing" distance metric
        printdetail: a flag to print some detail for debug
        fullcheck: Whether to consider the entire line for distance metrics or just the endpoints
        azfilter: whether to only check distance when the azimuth are aligned 
    Outputs:
        i,j,distance: The line indices and distance 
    '''
    #We can consider two cases of distance metric:
    #1) we filter lines with matching azimuth which overlap on some frame
    #2) we ignore azimuth and enable all frames, usually used only in run_astrometry.net for post-process track association. 

    if extracheck:
        extrap=True; azfilter=True
        dist=np.nan*np.zeros((cc,cc))
        for z1 in range(cc):
            locline = np.copy(goodlines_i[:6].squeeze())
            x1,y1 = interp_frame_xy(locline,z1,double=True,printdetail=printdetail,extrap=extrap)#,extrap=True)
            if not all(np.array([x1,y1])==0):
                for z2 in range(cc):
                    locline = np.copy(goodlines_j[:6].squeeze())
                    x2,y2 = interp_frame_xy(locline,z2,double=True,printdetail=printdetail,extrap=extrap)#,extrap=True)
                    if not all(np.array([x2,y2])==0):
                        disti=((x1-x2)**2. + (y1-y2)**2.)**.5 #L2 norm on the image frame 
                        dist[z1,z2] = disti #if (disti>(align_diam_j/2.)) else 0.

        #Also check endpoints explicitly 
        dvalA = np.linalg.norm(goodlines_i[:3]-goodlines_j[:3])
        dvalB = np.linalg.norm(goodlines_i[:3]-goodlines_j[3:6])
        dvalC = np.linalg.norm(goodlines_i[3:6]-goodlines_j[:3])
        dvalD = np.linalg.norm(goodlines_i[3:6]-goodlines_j[3:6])
        dvalEnd=min([dvalA,dvalB,dvalC,dvalD])

        #If distance is within diameter, it should be valid as a "matching" line. 
        dist[dist<=(align_diam_j/2.)] = 0 #vectorize 
        if np.isnan(dist).all():
            dval=np.inf
        else:
            dval = np.nanmin(dist)                  
        dval=min(dval,dvalEnd)
        return i,j,dval



    extrap = ~azfilter 

    if fullcheck: #align_diam_j>=0:
        #If checking the entire lines, find the minmum distance on any frame ('z-distance' ignored due to scale difference between pixels and frame count)
        dist=np.nan*np.zeros((cc+1,))
        for z in range(cc):
            locline = np.copy(goodlines_i[:6].squeeze())
            x1,y1 = interp_frame_xy(locline,z,double=True,printdetail=printdetail,extrap=extrap)#,extrap=True)
            if not all(np.array([x1,y1])==0):
                locline = np.copy(goodlines_j[:6].squeeze())
                x2,y2 = interp_frame_xy(locline,z,double=True,printdetail=printdetail,extrap=extrap)#,extrap=True)
                if not all(np.array([x2,y2])==0):
                    disti=((x1-x2)**2. + (y1-y2)**2.)**.5 #L2 norm on the image frame 
                    dist[z] = disti #if (disti>(align_diam_j/2.)) else 0.
        #Also check endpoints explicitly 
       
        if azfilter: #emergency needed in clean-up confirmation tests
            dvalA = np.linalg.norm(goodlines_i[:3]-goodlines_j[:3])
            dvalB = np.linalg.norm(goodlines_i[:3]-goodlines_j[3:6])
            dvalC = np.linalg.norm(goodlines_i[3:6]-goodlines_j[:3])
            dvalD = np.linalg.norm(goodlines_i[3:6]-goodlines_j[3:6])
            dvalEnd=min([dvalA,dvalB,dvalC,dvalD])
        else:
            dvalA = np.linalg.norm(goodlines_i[:3]-goodlines_j[:3])
            dvalD = np.linalg.norm(goodlines_i[3:6]-goodlines_j[3:6])
            dvalEnd=min(dvalA,dvalD)#min([dvalA,dvalB,dvalC,dvalD])

        dist[-1] = dvalEnd
        #If distance is within diameter, it should be valid as a "matching" line. 
        dist[dist<=(align_diam_j/2.)] = 0 #vectorize 
        if np.isnan(dist).all():
            dval=np.inf
        else:
            dval = np.nanmin(dist)
    else:
        #Only consider endpoints if needing faster runtime (ie during star removal before pre-cluster association)
        try:
            dvalA = np.linalg.norm(goodlines_i[:3]-goodlines_j[:3])
            dvalD = np.linalg.norm(goodlines_i[3:6]-goodlines_j[3:6])
            dval = min(dvalA,dvalD)
        except Exception as e:
            print('distmat_sub error in dvalA/dvalB')
            print(e)
      
    return i,j,dval #np.nanmin(dist)#np.linalg.norm(dist[np.isnan(dist)!=1])#np.nanmin(dist)

def remove_matches_block_par(ii,jinit,lines1loc,lines2loc,diamloc,zrange,septol,identical,fullcheck,azfilter,extracheck):
    '''
    Helper for remove_matches_block.  Runs distmat_sub on a partition of the full data
    Input:
        ii/jinit: The first index of the paritioned line set as a member of the entire line lists
        lines1loc/lines2loc: The lines as a partition of the entire data set
        diamloc: The diameters of the lines2loc elements
        zrange: The maximum number of frames the lines have z1<=z2<=zmax in 
        septo: How far apart endpoints may be to be considered for distance metric computation
        identical: Flag for if lines1/lines2 are the same data for match removal
        fullcheck: whether to condier any line intersection or just endpoint separation 
        azfilter: whether to condider if lines have a valid azimuth orientation alignment or not 
    Output:
        r0/r1list: indices of the full lines1/lines2 set of the r2list data
        r2list: the distance matrix for the line pair above 
    '''
    #If septol is provided, only consider those lines which have a first endpoint within some coarse distance 
    if not np.isnan(septol):
        sepMatA = sp.spatial.distance_matrix(lines1loc[:,:2],lines2loc[:,:2])<septol
        sepMatA = sepMatA.astype(dtype=bool)
        sepMat = sepMatA 
    else:
        sepMat = np.ones((lines1loc.shape[0],lines2loc.shape[0])).astype(dtype=bool)
    #If the line sets are identical, can use triangular access for faster speed
    if identical:
        sepMat = np.triu(sepMat) #elements below main diagonal zeroed (upper triangular)
        np.fill_diagonal(sepMat,0)
    #Iterate over the data, only considering valid elements for runtime 
    r0list=[]; r1list=[]; r2list=[]
    for sepIdx in np.argwhere(sepMat):
        i = sepIdx[0]
        j = sepIdx[1]
        try:
            _,_,r2 = distmat_sub(i,j,zrange,lines1loc[i,:],lines2loc[j,:],diamloc[j],False,fullcheck,azfilter,extracheck)
            r0list.append(i+ii); r1list.append(j+jinit); r2list.append(r2);        #distMat[r0,r1]=r2
        except Exception as e:
            continue
    return r0list,r1list,r2list 

def remove_matches_block(lines1, lines2,zrange,diam=[],identical=False,septol=np.nan,printdetail=False,runspliner=True,fullcheck=True,azfilter=True,disttol=None,extracheck=False): #lines1=goodlines_old, lines2=goodlines
    '''
    Check if lines are duplicates, or close enough to be considered similar members of a larger line.  
    Has flags for checking closest approach or just endpoints, for selecting the longest or merging lines directly, and filtering be orientation.
    Input:
        lines1: The [x1,y1,z1,x2,y2,z2,...] line set, usually of the entire original data set 
        lines2: The other line set is either associated lines (smaller list) or a duplicate list
        zrange: The maximum number of frames the lines have z1<=z2<=zmax in 
        diam: The diameters of the lines2 elements
        identical: Flag for if lines1/lines2 are the same data for match removal
        septo: How far apart endpoints may be to be considered for distance metric computation
        runspliner: whether to merge lines or just pick the longest nearby element
        fullcheck: whether to condier any line intersection or just endpoint separation 
        azfilter: whether to condider if lines have a valid azimuth orientation alignment or not 
    Output:
        lines1: The merged disjoint set
        extras: Those lines from lines1 which are not matched by lines2, usually only used in printing, or in detect_outliers_agg
    '''
    #if identical, we will have to iterate until no improvement, slow but effective 
    LOLD = 0
    LNEW = len(lines1)
    #Make sure z1<=z2 on each set
    lines1=fixlines(lines1)    
    lines2=fixlines(lines2)  
    if printdetail:
        print(lines1)
    while not LNEW==LOLD:
        if identical:
            lines2 = np.copy(lines1) #trick for removal 
        LOLD=np.copy(LNEW)
        #Get the size of the data being considered
        fx = len(lines1) #input
        fy = len(lines2) #reduced'
        #Build a matrix for storing information of matches 
        #TODO:This can crash is the lines are too big!
        distMat =np.nan*np.zeros((fx,fy))#
        if identical:
                np.fill_diagonal(distMat,np.inf)
        if len(diam)<fy:
            diam=-1*np.ones(fy) #default empty
       
        #If many lines, speed up via parallelizing partitions of the lines sets
        iterable=[]
        maxsize = 5000
        atime=time.time()#-atime
        iwindow=100 if identical else 1000
        atime2=time.time()#-atime
        iterable=[]
        for ii in range(0,fx,iwindow):
            iend = min(ii+iwindow,fx)
            jinit = ii if identical else 0 #put here for deflation, block is memflops not flops themselves, no parallel
            iterable.append([ii,jinit,lines1[ii:iend,:],lines2[jinit:,:],diam[jinit:],zrange,septol,identical,fullcheck,azfilter,extracheck])

        #Run parallelization, unless there are few enough lines that overhead would cost more
        process_count=int(min(len(iterable),np.floor(cpu_count()/4)))#int(numsteps)
        chunksize=int(np.ceil(len(iterable)/process_count)) #8*8=64, can't have that many parallel children!
        if chunksize>5:
            with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                results = pool.starmap(remove_matches_block_par,iterable,chunksize=chunksize)
        else:
            results = []
            for iti in iterable:
                results.append(remove_matches_block_par(iti[0],iti[1],iti[2],iti[3],iti[4],iti[5],iti[6],iti[7],iti[8],iti[9],iti[10]))
        for r in results:
            ilist = r[0]
            jlist = r[1]
            dlist = r[2]
            for ri in range(len(ilist)):
                distMat[ilist[ri],jlist[ri]]=dlist[ri]
        #print('diam',diam)
        #for di in range(distMat.shape[0]):
        #    print(distMat[di,:])
        #Save space for next step by reducing bit memory requirement via bool conversion 
        atime2=time.time()-atime2
        distMat[np.isnan(distMat)]=np.inf
        disttol = 1. if (disttol is None) else disttol #Edge case handling

        #print('distMatmin %.2f  / %.2f'%(np.amin(distMat), disttol))
        darr=np.where(distMat==np.amin(distMat))
        #print('distmatmin=',np.amin(distMat))
        #print(lines1[darr[0],:6])
        #print(lines2[darr[1],:6])

        distMat = distMat<disttol
        distMat = distMat.astype(dtype=bool)
   
        #If spliner, merge matching lines 
        atime=time.time()
        if runspliner:
            if printdetail:
                print('SPLINER:')#,end=' ')
            try:
                lines1,lines2,distMat = spliner(lines1,lines2,distMat,printdetail,azfilter=azfilter)
            except Exception as e:
                print('error calling spliner')
                print(e)
                quit()
            if printdetail:
                print('spliner done')
        atime=time.time()-atime
        atime=time.time()#-atime
        #print('splined:')
        #print(lines1[darr[0],:6])
        #print(lines2[darr[1],:6])

        #Now switch lines such that the longest matching line will be saved 
        kX= np.copy(np.linalg.norm(lines1[:,:3]-lines1[:,3:6],axis=1))
        kY= np.copy(np.linalg.norm(lines2[:,:3]-lines2[:,3:6],axis=1))
        whilemarker = 1
        whilecounter=1
        atime2=time.time()#-atime2
        while whilemarker:
            whilecounter+=1
            whilemarker=0
            #Replacing for loop with boolean vector indexing saves some time if very few elements
            distIdxList = np.argwhere(distMat)
            distIdxList = distIdxList[np.lexsort((distIdxList[:,0],distIdxList[:,1]))]
            for distIdx in distIdxList:#np.argwhere(distMat):
                i = distIdx[0]
                j = distIdx[1]
                if kX[i]>kY[j]: #if the old is longer than the new
                  templine = np.copy(lines1[i,:])
                  lines1[i,:] = np.copy(lines2[j,:])
                  lines2[j,:] = np.copy(templine)
                  tempk = np.copy(kX[i])
                  kX[i]=np.copy(kY[j])
                  kY[j]=np.copy(tempk)
                  whilemarker=1
                  #distMat does not need swapped since the matrix is boolean

        atime2=time.time()-atime2
        atime=time.time()-atime
        atime2=time.time()#-atime
        #Form the disjoint set merging lines 1 and lines 2
        if not identical:
            try:
                unmatched= np.asarray(np.any(~distMat,axis=1))
                if np.count_nonzero(unmatched)>0:
                    extras = lines1[unmatched, :]
                    lines1 = np.vstack([lines2, extras]) #merge the old unmatched and the new associated lines
                else:
                    extras=[]
                    lines1 = np.copy(lines2)
            except Exception as e:
                print(e)
                extras=[]
                lines1=np.copy(lines2)
                
        else:
            linestemp=[]
            #If using identical, we need to take some additional tricks to prevent reuse 
            usedvec = np.zeros(len(lines1)).astype(dtype=bool)
            dcount=0
            ndcount=0
            for i in range(len(lines1)):#enumerate(lines1):
                if not usedvec[i]:
                    jlist = np.argwhere(distMat[i,:])
                    if len(jlist)>0:
                        ndcount+=1
                        jlistidx = np.argmax(kY[jlist])
                        j = jlist[jlistidx] #passes distance and is longest
                        j = j if distMat[i,j] else i
                    else:
                        j = i
                    if j==i:
                        dcount+=1
                    distMat[i,:]=False
                    distMat[j,:]=False
                    distMat[:,i]=False
                    distMat[:,j]=False
                    usedvec[i]=True
                    usedvec[j]=True
                    linestemp.append(lines2[j])
          
            lines1 = np.vstack(linestemp)
            extras=[]
        atime2=time.time()-atime2

        LNEW = len(lines1)
        #End if not identical, or if there was no improvement
        if not identical: 
            LOLD=LNEW
    return lines1,extras


