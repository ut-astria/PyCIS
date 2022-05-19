'''
PyCIS - Python Computational Inference from Structure

pylib/main_pipe.py: Pipeline for PyCIS processing 

TODO:
Star filtering before clustering should be updated to account for possible sidereal-tracking modes, with possible tracking noise
A-priori filtering before outlier detection (Hot pixels/IOD) should be revisted for sidereal-tracking mode
Handling of recursive-conditioning "small N_T" paradox in outlier detection


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
# Import other python functions (mod_fits FITS header update function temporarily disabled)
from pylib.import_fits import import_fits, import_bin #, mod_fits
from pylib.run_pycis_lsd import run_pycis_lsd
from pylib.detect_outliers import detect_outliers
from pylib.detect_outliers_agg import detect_outliers as detect_outliers_new
from pylib.detect_outliers_agg import outlier_alg
from pylib.print_detections import print_detections, print_detections_clust, print_detections_window
from pylib.run_astrometry import run_astrometry
#Util functions
from pylib.detect_utils import my_filter, remove_matches_block
from pylib.main_utils import  associate_multiscale, apriori_filter_star_frames_adaptwide, apriori_filter





def run_pycis(
    satfolder,satlist,datatype,numsteps,
    imgfolder,tdmfolder,imgname,vs,makegif,printcluster,
    solvemarkov,resolve,printonly,
    imscale,framerange,a,tc,median,shift=0,sig=0,e2=0,makeimg=1,solveastro=1,
    linename='NULL',binfactor=1,fps=5,tle=[],imgastro=0,cluster=1,background=0,
    injectScale=0, bigparallel=0, subprocess_count=10,runoutlier=1,
    getmedianimg=0,bigmedianimg=None,minimg=0,p1=None,p2=None):
    '''
    Main interface for processing a window of data
    For large data sets, this function provides functionality for Line Segment Detection, 
    to get the first-order detections from the dense optical data. 
    End_pipe should be used for clustering and outlier detection on joined windows. 

    Input: data/...
        yyymmdd_norad_satname/*.fit - a folder with raw fits frames
        SEE runpycis.py FOR INPUT VARIABLE DETAILS 

    Numeric Output:
        results/imgfolder_work/... 
            data1_name.npy - edge line detections
            data2_name.npy - center line detections
            goodlines_name.npy - 2nd-order meaningful detections
            badlines_name.npy - 1st-order meaningful only detections
            img_name.npy - last fit frame data with projected detections (2nd order in red)
            vidAll_name.npy - animated fits frame data with projected detections over time (2nd order in red)
            vidObj_name.npy - animated fits frame data with only 2nd-order projected detections over time 
        results/imgfolder/...
            img*A[a]B[b]*png / img*ASSOC*png - still image showing the detected objects, projected onto a single frame
                                AB indicate window-based frame range when considering a large number of frames, necessary for the LSD algorithm
                                ASSOC indicates the result after clustering/outlier detection on the unified set of frames 
            videoAll*A[a]B[b]*gif / videoObj*ASSOC*gif - videos showing detections on an animation of all frames (a contrario results, before astrometric calibration)
            ASTRO*ASSOC*gif - videos showing detections on an animation of all frames (after astrometric calibration, uses source extraction to help fit extrapolated trajectories to non-linear paths)
                                note: here, positive detections (strong fit to a-contrario detection) in red, rejected noise in blue 
            *ASSOC*FINALVOL.html - Interactive 3D plot of the a-contrario detections (old format)
            *ASSOC*FINALVOL.html - Interactive 3D plot of a-contrario detections and astrometric calibration for improved tracking and noise rejection (new format)
        tdmfolder/...
            *xml - the CCSDS-format Tracking Data Message for the data set, astrometry using Astrometry.net and Source Extractor (SEP for python) for improving precision 

    '''

    ## BEGIN TIMING 
    starttime=time.time()
    numtests =1
    outfolder='%s_work'%imgfolder
    scale=1. #Gaussian downsampling volume ratio 
    sigma=.6 #Gaussian downsampling deviation factor


    #Select data from folder - loopable 
    for satidx, satname in enumerate(satlist):
        #prename='%s_%s'%(imgname,satname)
        prename='%s_%s'%(imgname,satname)
        if linename=='NULL':
            tempname=prename
        else:
            tempname='%s_%s'%(linename,satname)
        if os.path.exists('%s/record_%s.json'%(outfolder,tempname)) and cluster==2 and makeimg==0:
            continue

        input_dir = '%s/%s'%(satfolder,satname)
        print('kernel in:',(p1,p2))
        if datatype=='bin': 
            I3, headers = import_bin(input_dir, savedata=0, subsample=1, 
                framerange=framerange, scale=imscale,median=median,shift=shift)
        elif datatype=='fits':
            printheader=0
            if getmedianimg==1:
                I3, headers,minimg = import_fits(input_dir,savedata=0,subsample=1,
                    framerange=[-1,-1],scale=imscale,printheader=0,gauss=[scale,sigma],median=0,
                    shift=shift,sig=sig,binfactor=binfactor, 
                    bigmedian=0,minimg=np.nan)
                I3 = np.asarray(I3)
                bigmedianimg = np.copy(np.median(I3,axis=0))
                I3 = I3-bigmedianimg[np.newaxis,:,:]
                I3 = I3-minimg+1.
                I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))
                linetime = time.time()
                print('bigI3 domain: ', (I3.min(), I3.max()))
                p1,p2 = run_pycis_lsd(
                    I3, outfolder, prename,
                    numsteps,bigparallel,
                    solvemarkov,1,printonly,
                    a,tc,returnmarkov=True) #resolve=1 when desired
                print('kernel out:',(p1,p2))
                print('LSD MARKOV-ONLY RUNTIME: %.2f sec (%.2f min)\n\n'%(linetime,linetime/60.))
                I3, headers, fullsize = import_fits(input_dir,savedata=0,subsample=1,
                    framerange=framerange,scale=imscale,printheader=0,gauss=[scale,sigma],median=median,
                    shift=shift,sig=sig,binfactor=binfactor, 
                    bigmedian=1,bigmedianimg = bigmedianimg,minimg=np.copy(minimg),returnsize=True)
            elif getmedianimg==2:
                I3, headers, fullsize = import_fits(input_dir,savedata=0,subsample=1,
                    framerange=framerange,scale=imscale,printheader=0,gauss=[scale,sigma],median=median,
                    shift=shift,sig=sig,binfactor=binfactor, 
                    bigmedian=1,bigmedianimg = bigmedianimg,minimg=minimg,returnsize=True)
            else:
                I3, headers, fullsize = import_fits(input_dir, savedata=0, subsample=1, 
                    framerange=framerange, scale=imscale, printheader=printheader,gauss=[scale,sigma],median=median,
                    shift=shift,sig=sig,binfactor=binfactor,returnsize=True)

        #I3 = np.load('%s/%s.npy'%(satfolder,satname))
        I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))

        if printonly==1:
            print_detections(np.copy(I3),[],[],folder=imgfolder,savename=prename, makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1,background=background)
            continue


        ## FIRST-ORDER A_CONTRARIO ANALYSIS

        linetime = time.time()
        lines = run_pycis_lsd(
            I3, outfolder, prename,
            numsteps,bigparallel,
            solvemarkov,resolve,printonly,
            a,tc,p1in=np.copy(p1),p2in=np.copy(p2),fullsize=fullsize)
        linetime = time.time()-linetime
        print('LSD RUNTIME: %.2f sec (%.2f min)\n\n'%(linetime,linetime/60.))

        np.set_printoptions(suppress=True)
        
        ## SECOND-ORDER  A_CONTRARIO ANALYSIS

        if linename=='NULL':
            tempname=prename
        else:
            tempname='%s_%s'%(linename,satname)


        if cluster==1 or cluster==2:

            clustertime=time.time()
            linesexist=os.path.exists('%s/goodlines_%s.npy'%(outfolder,tempname)) and (resolve==0)
            if linesexist and cluster==2:
                goodlines = np.load("%s/goodlines_%s.npy"%(outfolder,tempname))
                badlines = np.load("%s/badlines_%s.npy"%(outfolder,tempname))
            else:
                if injectScale==0:
                    goodlines, badlines = detect_outliers(I3.shape,lines,folder=outfolder,
                        savename=tempname,e2=e2,subprocess_count=subprocess_count)
                else:
                    avoidrun=False
                    if runoutlier==0: #if no association or outliers, quick escape to assume all-window analysis
                        avoidrun=True
                    goodlines, badlines = detect_outliers_new(I3.shape,lines,folder=outfolder,
                        savename=tempname,e2=e2,injectScale=injectScale,subprocess_count=subprocess_count,
                        postcluster=runoutlier,runoutlier=runoutlier,avoidrun=avoidrun)

            clustertime=time.time()-clustertime
            print('CLUSTER RUNTIME: %.2f sec (%.2f min)\n\n'%(clustertime,clustertime/60.))

            #PRINT RESULTS
            if not makeimg==0:
                if printcluster==0:
                    print_detections(np.copy(I3),goodlines,badlines,folder=imgfolder,savename=tempname, 
                        makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1,background=background)
                elif printcluster==1:
                    print_detections_clust(np.copy(I3),outfolder,tempname,folder=imgfolder,savename=tempname, 
                        makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1)
                else:
                    print_detections(np.copy(I3),goodlines,badlines,folder=imgfolder,savename=tempname,
                        makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1,background=background)
                    print_detections_clust(np.copy(I3),outfolder,tempname,folder=imgfolder,savename=tempname, 
                        makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1)

            # PROCESS ASTROMETRY
            astrotime=time.time()
            jsonexists = os.path.exists('%s/record_%s.json'%(outfolder,tempname))
            if solveastro==1 or (solveastro==2 and jsonexists==0):
                makejson=1
                aa,bb,cc = I3.shape
                headersnew = run_astrometry(I3,goodlines, badlines, headers, scale=(aa,bb), 
                    folder=outfolder,tdmfolder=tdmfolder,savename=tempname,makejson=makejson,tle=tle,
                    binfactor=binfactor,imgastro=imgastro,subprocess_count=subprocess_count,imgshape=I3.shape)
            astrotime=time.time()-astrotime
            print('ASTRO RUNTIME: %.2f sec (%.2f min)\n\n'%(astrotime,astrotime/60.))

        '''
        ## RUN ASTROMETRY AND UPDATE HEADERS
        newfits = 'new%s/%s'%(satfolder,satname)
        if not os.path.exists('new%s'%satfolder):
            os.makedirs('new%s'%satfolder)
        if not os.path.exists(newfits):
            os.makedirs(newfits)
        mod_fits(input_dir, headersnew, folder=newfits,
            subsample=1, framerange=framerange)
        '''
    ## PLOT TIMING DATA 
    avgtime = (time.time()-starttime)/1. #numtests
    print('TOTAL RUNTIME: %.2f sec (%.2f min)\n\n'%(avgtime,avgtime/60.))
    if getmedianimg==1:
        return bigmedianimg,minimg,p1,p2
    return 1,minimg,p1,p2

def end_pipe(satfolder,satlist,datatype,numsteps,
    allname,alllinename, imgnamelist, linenamelist,framerangelist,
    imgfolder,tdmfolder,vs,makegif,printcluster,
    resolve,imscale, median,shift,sig=0,e2=0,makeimg=1,solveastro=1,
    linename='NULL',binfactor=1,fps=5,tle=[],imgastro=0,cluster=1,background=0,
    runoutlier=0, subprocess_count=10):
    '''
    Interface for processing data after windowed LSD analysis,
    this includes Clustering and Outlier detection stages.  
    For computational effiacy, star-like features are first removed by considering length-weighted orientation. 
    We regularly check for duplicate lines to handle edge cases in track association. 
    To prevent removal of untracked objects, we first perform partial track association before clustering, 
    followed by full association using multiple scales.  
    Before outlier detection, we filter for hot pixels, invalid measurements (unable to perform IOD), and associated stars.
    Results are fed to the astrometry pipeline which enables refined non-linear tracking 
    using local kernel matching on extrapolated a-contrario results. 

    Input: data/...
        yyymmdd_norad_satname/*.fit - a folder with raw fits frames
        SEE runpycis.py FOR INPUT VARIABLE DETAILS 

    Numeric Output:
        results/imgfolder_work/... 
            data1_name.npy - edge line detections
            data2_name.npy - center line detections
            goodlines_name.npy - 2nd-order meaningful detections
            badlines_name.npy - 1st-order meaningful only detections
            img_name.npy - last fit frame data with projected detections (2nd order in red)
            vidAll_name.npy - animated fits frame data with projected detections over time (2nd order in red)
            vidObj_name.npy - animated fits frame data with only 2nd-order projected detections over time 
        results/imgfolder/...
            img*A[a]B[b]*png / img*ASSOC*png - still image showing the detected objects, projected onto a single frame
                                AB indicate window-based frame range when considering a large number of frames, necessary for the LSD algorithm
                                ASSOC indicates the result after clustering/outlier detection on the unified set of frames 
            videoAll*A[a]B[b]*gif / videoObj*ASSOC*gif - videos showing detections on an animation of all frames (a contrario results, before astrometric calibration)
            ASTRO*ASSOC*gif - videos showing detections on an animation of all frames (after astrometric calibration, uses source extraction to help fit extrapolated trajectories to non-linear paths)
                                note: here, positive detections (strong fit to a-contrario detection) in red, rejected noise in blue 
            *ASSOC*FINALVOL.html - Interactive 3D plot of the a-contrario detections (old format)
            *ASSOC*FINALVOL.html - Interactive 3D plot of a-contrario detections and astrometric calibration for improved tracking and noise rejection (new format)
        tdmfolder/...
            *xml - the CCSDS-format Tracking Data Message for the data set, astrometry using Astrometry.net and Source Extractor (SEP for python) for improving precision 

    '''
    ## BEGIN TIMING 
    np.set_printoptions(suppress=True)

    starttime=time.time()
    numtests =1
    outfolder='%s_work'%imgfolder
    scale=1. #Gaussian downsampling volume ratio 
    sigma=.6 #Gaussian downsampling deviation factor
    skip=True
    I3loc_all = []
    headers_all = []
    goodlines_all = []
    badlines_all = []
    lastrange=np.asarray(framerangelist[0])
    

    #IMPORT DATA FROM THE WINDOWED PROCESSES
    offset=0
    headerflag=True  
    for satidx, satname in enumerate(satlist):
        for frameidx, framerange in enumerate(framerangelist):
            if offset==0 and frameidx>0:
                headerflag=False
            offset = max(0,lastrange[1]-framerange[0])
            lastrange = np.copy(framerange)
            if offset>=(framerange[1]-framerange[0]):
                offset=0
            imgname=imgnamelist[frameidx]
            prename='%s_%s'%(imgname,satname)
            linename=linenamelist[frameidx]
            if linename=='NULL':
                tempname=prename
            else:
                tempname='%s_%s'%(linename,satname)
            if os.path.exists('%s/record_%s.json'%(outfolder,tempname)) and cluster==2 and makeimg==0:
                #print('record found for %s'%tempname)
                continue

            input_dir = '%s/%s'%(satfolder,satname)
            #print('imporing data for %s '%satname)
            skip = (frameidx+1)!=len(framerangelist)
            #print('SKIPPER: ',skip)
            I3loc_all.append(input_dir)
            if datatype=='bin':
                I3, headers = import_bin(input_dir, savedata=0, subsample=1,
                    framerange=framerange, scale=imscale,median=median,shift=shift)
            elif datatype=='fits':
                printheader=0
                if frameidx==0:
                    _, headers_all = import_fits(input_dir, savedata=0, subsample=1,
                        framerange=[-1,-1], scale=imscale, printheader=printheader,gauss=[scale,sigma],median=median,shift=shift,sig=sig,binfactor=binfactor,skip=skip)
                I3, _ = import_fits(input_dir, savedata=0, subsample=1,
                    framerange=framerange, scale=imscale, printheader=printheader,gauss=[scale,sigma],median=median,shift=shift,sig=sig,binfactor=binfactor,skip=skip)
            if not skip:
                #print('storing image as contiguous array...',flush=True)
                I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))
            #offset = np.abs(lastrange[0]-framerange[0])
            if linename=='NULL':
                tempname=prename
            else:
                tempname='%s_%s'%(linename,satname)
            #print('imporing lines for %s '%tempname)
            clustertime=time.time()
            linesexist=os.path.exists('%s/goodlines_%s.npy'%(outfolder,tempname))
            if not linesexist:
                print('ERROR: associate_all() requires clustering to have been completed on a windowed set.  Please run main.')
                quit()
            goodlines = np.load("%s/goodlines_%s.npy"%(outfolder,tempname))
            badlines = np.load("%s/badlines_%s.npy"%(outfolder,tempname))
            #CORRECT FOR FRAME OFFSETS
            goodlines[:,[2,5]] = goodlines[:,[2,5]] + framerange[0]
            badlines[:,[2,5]] = badlines[:,[2,5]] + framerange[0]
            goodlines_all.append(goodlines)
            badlines_all.append(badlines)

        #CONCATENATE WINDOW RESULTS
        goodlines = np.vstack(goodlines_all)
        badlines = np.vstack(badlines_all)
        
        print('\n\nGLOBAL LINE COUNT',flush=True)
        print('good/bad = ',[len(goodlines),len(badlines)])

        #Define name and size varaibles
        prename='%s_%s'%(allname,satname)
        if linename=='NULL':
                tempname=prename
        else:
                tempname='%s_%s'%(alllinename,satname)
        fullshape=[int(4096/binfactor),int(4096/binfactor),int(framerange[1]-framerange[0]+1)]
        print('FRAME SHAPE: ',fullshape)
        aa,bb,cc = fullshape
        cc2 = len(headers_all)
        print('confirming cc2:',cc2)
        fullshape = [aa,bb,cc2]
        print('NEW SHAPE: ',fullshape)
        print('CONFIRM LINES IN RANGE',flush=True)
        goodlines=my_filter(goodlines,fullshape)
        badlines=my_filter(badlines,fullshape)
        print('good/bad = ',[len(goodlines),len(badlines)])

        ## RUN ASSOCIATION (SECOND/THIRD ORDER A_CONTRARIO ANALYSIS)
        clustertime=time.time()
        linesexist=os.path.exists('%s/goodlines_%s.npy'%(outfolder,tempname))
        print('Beginning cluster work',flush=True) 
        if resolve or (cluster==1 or (not linesexist)):

            ## FIRST-ORDER DETECTION SET NEEDS CLEANED TO MITIGATE CURRENT LSD IMPLEMENTATION ERRORS, 
            # COPIES FROM WINDOWED ANALYSIS (memory constraint), REDUCING LINE DENTITY (runtime and memory constraint)
            # AND ASSOCIATING LINES TO PREVENT REMOVAL OF UNTRACKED OBJECTS (temporal aliasing / sensor constraint)
            #'''
            #Ensure data is properly in range
            print('pre-cluster apriori filtering...',flush=True)
            lines = np.vstack((goodlines,badlines))
            linesIn=np.copy(lines)
            linesIdx = linesIn[:,2]>linesIn[:,5]
            print('fixing %d lines'%np.count_nonzero(linesIdx))
            lines[linesIdx,0:3] = np.copy(linesIn[linesIdx,3:6])
            lines[linesIdx,3:6] = np.copy(linesIn[linesIdx,0:3])
            del linesIn
            lines = my_filter(lines,[aa,bb,cc2],skipper=False, buffer=1.)
            print('newshape after filter with buffer, ',len(lines))

            #Remove dominant star features, assume a-priori that these are some dominant x-y behavior, weight azimuth by length 
            #TODO: Need to check if dominant energy would rather this be in a star-staring direction, need pipeline flag.
            startime=time.time()
            goodlines,med_az,starlines = apriori_filter_star_frames_adaptwide(np.copy(lines),cc2,0.05*min(aa,bb),med_az=None,getstars=True)
            print('STAR_FILTERED LINE COUNT: %d'%len(goodlines),flush=True)
            startime = time.time()-startime
            print('STARTIME IN %.2f'%(startime/60.))
            #And make sure that any 'duplicate lines' of overlapping windows are removed
            goodlines = my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)
            goodlines,_ = remove_matches_block(goodlines, goodlines,cc2,identical=True,septol=0.05*min(aa,bb)) #lines1=goodlines_old, lines2=goodlines
            print('REMOVE MATCHES: ',len(goodlines))
           
            #Use multi-scale a-contrario line detection to merge lines, avoid filtereing out Untracked objects as clusters 
            #First associate near-horizontal lines (image-plane)
            print('pre-cluster associations...',flush=True)
            testertime=time.time()#-clustertime
            try:
                goodlines,_p = associate_multiscale(9,np.copy(goodlines),fullshape,outfolder,tempname,depth=2,window=cc,filt='XY')
            except Exception as e:
                print(e)
            testertime=time.time()-testertime#-clustertime
            print('XY-ASSOC TAKES %.2f min'%(testertime/60.))
            goodlines = my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)
            testertime=time.time()#-cl
            #Then associate in the vertical (time) direction 
            try:
                goodlines,_ = associate_multiscale(9,np.copy(goodlines),fullshape,outfolder,tempname,depth=2,window=cc,filt='Z')
            except Exception as e:
                print(e)
            testertime=time.time()-testertime#-clustertime
            print('Z-ASSOC TAKES %.2f min'%(testertime/60.))
            #copies will be safelty removed 
            goodlines = my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)
            print('PRECLUSTER ASSOCIATE COUNT: ',len(goodlines))
            goodlines,_ = remove_matches_block(goodlines, goodlines,cc2,identical=True,septol=0.05*min(aa,bb)) #lines1=goodlines_old, lines2=goodlines
            print('REMOVE MATCHES: ',len(goodlines))
            #goodlines = apriori_filter_star(np.copy(lines)) 
            ## SECOND-ORDER A CONTRARIO ANALYSIS: REMOVE CLUSTERS OF LINES 
            print('clustering...',flush=True)
            newlinesexist=os.path.exists('%s/goodtestoutliers_%s.npy'%(outfolder,tempname))
            if resolve or (not newlinesexist): #runoutlier==1:
                print('SOVLING CLUSTERS')
                #print('CORRECTING DETECTIONS',flush=True)
                goodlines, rejlines = detect_outliers_new(fullshape,goodlines,folder=outfolder,
                    savename=tempname,e2=e2,injectScale=-1,subprocess_count=subprocess_count,
                    postcluster=0,runoutlier=0)
                badlines = rejlines#np.vstack([badlines,rejlines])
                goodlines=my_filter(goodlines,fullshape)
                badlines=my_filter(badlines,fullshape)
                print('filter good/bad = ',[len(goodlines),len(badlines)])
                np.save("%s/goodtestoutliers_%s.npy"%(outfolder,tempname),goodlines)
                np.save("%s/badtestoutliers_%s.npy"%(outfolder,tempname),badlines)
            else:
                print('LOADING CLUSTERS')
                goodlines = np.load("%s/goodtestoutliers_%s.npy"%(outfolder,tempname))
                badlines = np.load("%s/badtestoutliers_%s.npy"%(outfolder,tempname))
            #'''
            goodlines = np.load("%s/goodtestoutliers_%s.npy"%(outfolder,tempname))
            badlines = np.load("%s/badtestoutliers_%s.npy"%(outfolder,tempname))
            print('TEMP CLUSTERING ALL IN CORRECTION POSTCLUSTER')
            #print_detections_window(np.copy(I3),np.copy(goodlines),[],folder=imgfolder,savename='CLUSTER',#%s'%tempname,
            #    makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1,background=background,
            #    ccopt=len(headers_all), I3loc = I3loc_all, frames=framerangelist,
            #    imscale=imscale, median=median, shift=shift, datatype=datatype,binfactor=binfactor)
            ## SECOND GESTALT IMPLEMENTATION CLEAN-UP, ANOTHER ROUND OF ASSOCIATION AND COPY REMOVAL
            print('post-cluster associations...',flush=True)
            #Between cluster/outlier do: association, star track removal at any elevation, hot pixel and IOD removal
            goodlines = my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)
            goodlines,_ = associate_multiscale(5,np.copy(goodlines),fullshape,outfolder,tempname)
            print('POSTCLUSTER ASSOCIATE COUNT: ',len(goodlines))
            goodlines = my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)
            goodlines,_ = remove_matches_block(goodlines, goodlines,cc2,identical=True,septol=0.05*min(aa,bb),azfilter=True,disttol=0.001*min(aa,bb),extracheck=True) #lines1=goodlines_old, lines2=goodlines
            print('REMOVE MATCHES: ',len(goodlines))

            ## PREPARE THIRD GESTALT: EXPLOIT INFERED INFORMATION, hot pixels and IOD-invalid lines 
            #TODO: Hot pixels may conflict with a star-staring mode! Will need to prevent this in that case...
            #      or perhapse not, it would remove all stars in that case and help cluster implementation 
            print('post-cluster apriori filtering..',flush=True)
            goodlines = apriori_filter(np.copy(goodlines)) 
            goodlines = my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)

            ## THIRD GESTALT: OUTLIER DETECTION FOR ISOLATING ASOs IN NOISE
            #TODO: This will face a major issue when the conditioning is too strong that "everything is meaningful", and remove all lines.
            #      currently, if there are less than 3 lines needed for PCA, all are returned
            if runoutlier==1:
                print('RUNNING OUTLIER DETECTION ON FULL SET')
                lines = np.copy(goodlines) #includes associated and unassociate "non-clustered" data
                k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
                el = np.arccos((lines[:,5]-lines[:,2])/k_len)
                az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
                linesNorm = np.array([k_len*np.cos(az)*np.sin(el),k_len,lines[:,6],
                            k_len*np.sin(az)*np.sin(el),k_len*np.cos(el),lines[:,-1]]).T
                print('inputing %d lines'%len(linesNorm))
                print('outlier detection..',flush=True)
                goodidxfull,badidxfull = outlier_alg(linesNorm, 3,3, 0, runoutlier,name=tempname,folder=imgfolder)
                idxlist = np.zeros((len(lines)))
                idxlist[goodidxfull]=1

                goodlines_cluster = lines[idxlist==1]
                badlines_cluster = lines[idxlist==0]
                goodlines = goodlines_cluster
                print('good_cluster ',len(goodlines))
                print('bad_cluster ',len(badlines_cluster))
                badlines = np.vstack([badlines,badlines_cluster])
                #print_detections_window(np.copy(I3),np.copy(goodlines),np.copy(badlines_cluster),folder=imgfolder,savename='OUTLIER',#%s'%tempname,
                #    makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1,background=background,
                #    ccopt=len(headers_all), I3loc = I3loc_all, frames=framerangelist,
                #    imscale=imscale, median=median, shift=shift, datatype=datatype,binfactor=binfactor)
                print('badlines  ',len(badlines))
                goodlines=my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)
                badlines=my_filter(badlines,[aa,bb,cc2],skipper=False, buffer=1.)
                print('filter good/bad = ',[len(goodlines),len(badlines)])

                #FINALLY, remove those lines for which association made a non-outlier star association 
                print('Now removing persisting parallel/antiparallel star-like features ')
                if len(goodlines)>0:
                    goodlines,_ = apriori_filter_star_frames_adaptwide(np.copy(goodlines),cc2,0.05*min(aa,bb),med_az=med_az)
                    print('post-outlier association..',flush=True)
                    #goodlines,_ = apriori_filter_star(np.copy(goodlines),med_az=med_az) 
                    goodlines,_ = associate_multiscale(5,np.copy(goodlines),fullshape,outfolder,tempname)
                    goodlines=my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)
                    goodlines,_ = remove_matches_block(goodlines, goodlines,cc2,identical=True,septol=0.05*min(aa,bb))#,printdetail=True) #lines1=goodlines_old, lines2=goodlines
                    goodlines=my_filter(goodlines,[aa,bb,cc2],skipper=False, buffer=1.)

            #WITH A RECDUCED SET, NOW ENABLE VERTICLE ASSOCIATION
            print('goodlines:\n',goodlines)
            print("saving to... %s/goodlines_%s.npy"%(outfolder,tempname),flush=True)
            np.save("%s/goodlines_%s.npy"%(outfolder,tempname),goodlines)
            np.save("%s/badlines_%s.npy"%(outfolder,tempname),badlines)
            np.save("%s/starlines_%s.npy"%(outfolder,tempname),starlines)

        else:
            goodlines =np.load("%s/goodlines_%s.npy"%(outfolder,tempname))
            badlines = np.load("%s/badlines_%s.npy"%(outfolder,tempname))
            starlines = np.load("%s/starlines_%s.npy"%(outfolder,tempname))

        clustertime=time.time()-clustertime
        print('CLUSTER+OUTLIER TIME: %.2f sec (%.2f min)\n\n'%(clustertime,clustertime/60.),flush=True)

        #'''
        if not makeimg==0:
            if not printcluster==0:
                print('WARNING: print clustering not enabled.  setting printcluster=0')
                printcluster=0

            if printcluster==0:
                print_detections_window(np.copy(I3),np.copy(goodlines),[],folder=imgfolder,savename=tempname,
                    makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1,background=background,
                    ccopt=len(headers_all), I3loc = I3loc_all, frames=framerangelist,
                    imscale=imscale, median=median, shift=shift, datatype=datatype,binfactor=binfactor)
            elif printcluster==1:
                print_detections_clust(np.copy(I3),outfolder,tempname,folder=imgfolder,savename=tempname,
                    makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1)
            else:
                print_detections(np.copy(I3),goodlines,badlines,folder=imgfolder,savename=tempname,
                    makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1,background=background)
                print_detections_clust(np.copy(I3),outfolder,tempname,folder=imgfolder,savename=tempname,
                    makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1)
        #'''
        astrotime=time.time()
        jsonexists = os.path.exists('%s/record_%s.json'%(imgfolder,tempname))
        if solveastro==1 or (solveastro==2 and jsonexists==0):
            print('creating kernel:')
            print('starting astr0...')
            makejson=1
            headersnew = run_astrometry(I3,goodlines, starlines, headers_all, scale=(aa,bb),
                folder=imgfolder,tdmfolder=tdmfolder,savename=tempname,makejson=makejson,tle=tle,
                binfactor=binfactor,vs=vs,imgastro=imgastro,subprocess_count=subprocess_count,I3loc=I3loc_all,frames=framerangelist,imscale=imscale,median=median,shift=shift,datatype=datatype,imgshape=fullshape)
        astrotime=time.time()-astrotime
        print('ASTRO RUNTIME: %.2f sec (%.2f min)\n\n'%(astrotime,astrotime/60.))

        '''
        ## RUN ASTROMETRY AND UPDATE HEADERS
        newfits = 'new%s/%s'%(satfolder,satname)
        if not os.path.exists('new%s'%satfolder):
            os.makedirs('new%s'%satfolder)
        if not os.path.exists(newfits):
            os.makedirs(newfits)
        mod_fits(input_dir, headersnew, folder=newfits,
            subsample=1, framerange=framerange)
        '''
        
    ## PLOT TIMING DATA 
    avgtime = (time.time()-starttime)/1. #numtests
    print('TOTAL RUNTIME: %.2f sec (%.2f min)\n\n'%(avgtime,avgtime/60.))

