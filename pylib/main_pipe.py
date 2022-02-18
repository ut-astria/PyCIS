'''
PyCIS - Python Computational Inference from Structure

pylib/main_pipe.py: Pipeline for PyCIS processing 

TODO:
  Integrate run_astrometry (TDM), print_detections (html), and detect_outliers_agg (new clustering) updates

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
import json
import time
import scipy as sp
import scipy.stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Import other python functions 
from pylib.import_fits import import_fits, import_bin, mod_fits
from pylib.detect_outliers import detect_outliers
from pylib.detect_outliers_agg import detect_outliers as detect_outliers_new
from pylib.detect_outliers_agg import associate_lines, outlier_alg

#from pylib.newdetect import detect_outliers
from pylib.print_detections import print_detections, print_detections_clust, print_detections_window
from pylib.run_astrometry import run_astrometry
#blockprocessor
from multiprocessing import Pool, cpu_count,get_context,set_start_method
from functools import partial
from pylib.run_pycis_lsd import run_pycis_lsd


def run_pycis(
    satfolder,satlist,datatype,numsteps,
    imgfolder,imgname,vs,makegif,printcluster,
    solvemarkov,resolve,printonly,
    imscale,framerange,a,tc,median,shift,sig=0,e2=0,makeimg=1,solveastro=1,
    linename='NULL',binfactor=1,fps=5,tle=[],imgastro=0,cluster=1,background=0,
    injectScale=0, bigparallel=0, subprocess_count=10,runoutlier=1):
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

        #print('USING FAKE IMG')
        #I3 = np.zeros((1536,1536,30))
        #headers=[]
        #'''
        #prename='%s'%(imgname)
        input_dir = '%s/%s'%(satfolder,satname)
        if datatype=='bin': 
            I3, headers = import_bin(input_dir, savedata=0, subsample=1, 
                framerange=framerange, scale=imscale,median=median,shift=shift)
        elif datatype=='fits':
            printheader=0
            I3, headers = import_fits(input_dir, savedata=0, subsample=1, 
                framerange=framerange, scale=imscale, printheader=printheader,gauss=[scale,sigma],median=median,shift=shift,sig=sig,binfactor=binfactor)
        #I3 = np.load('%s/%s.npy'%(satfolder,satname))
        I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))
        #'''

        if printonly==1:
            print_detections(np.copy(I3),[],[],folder=imgfolder,savename=prename, makevid=0,makegif=makegif,vs=vs,fps=fps,amp=1,background=background)
            continue

        linetime = time.time()
        lines = run_pycis_lsd(
            I3, outfolder, prename,
            numsteps,bigparallel,
            solvemarkov,resolve,printonly,
            a,tc)
        linetime = time.time()-linetime
        print('LSD RUNTIME: %.2f sec (%.2f min)\n\n'%(linetime,linetime/60.))


        np.set_printoptions(suppress=True)
        
        ## SECOND NFA APPLICATION (SAVES LINE DATA)

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
                    goodlines, badlines = detect_outliers_new(I3.shape,lines,folder=outfolder,
                        savename=tempname,e2=e2,injectScale=injectScale,subprocess_count=subprocess_count,
                        postcluster=1,runoutlier=runoutlier)
            clustertime=time.time()-clustertime
            print('CLUSTER RUNTIME: %.2f sec (%.2f min)\n\n'%(clustertime,clustertime/60.))

            #print('quitting on line 599 after detect_outliers in demojupyter_noise')
            #quit()

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

            astrotime=time.time()
            jsonexists = os.path.exists('%s/record_%s.json'%(outfolder,tempname))
            if solveastro==1 or (solveastro==2 and jsonexists==0):
                makejson=1
                aa,bb,cc = I3.shape
                headersnew = run_astrometry(I3,goodlines, badlines, headers, scale=(aa,bb), 
                    folder=outfolder,savename=tempname,makejson=makejson,tle=tle,
                    binfactor=binfactor,imgastro=imgastro,subprocess_count=subprocess_count)
            astrotime=time.time()-astrotime
            print('ASTRO RUNTIME: %.2f sec (%.2f min)\n\n'%(astrotime,astrotime/60.))


        '''
        ## RUN ASTROMETRY AND UPDATE HEADERS
        aa,bb,cc = I3.shape
        headersnew = run_astrometry(goodlines, badlines, headers, scale=(aa,bb), folder=outfolder,savename=prename)
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


def end_pipe(satfolder,satlist,datatype,numsteps,
    allname,alllinename, imgnamelist, linenamelist,framerangelist,
    imgfolder,vs,makegif,printcluster,
    resolve,imscale, median,shift,sig=0,e2=0,makeimg=1,solveastro=1,
    linename='NULL',binfactor=1,fps=5,tle=[],imgastro=0,cluster=1,background=0,
    runoutlier=0, subprocess_count=10):

     ## BEGIN TIMING 
    starttime=time.time()
    numtests =1
    outfolder='%s_work'%imgfolder
    scale=1. #Gaussian downsampling volume ratio 
    sigma=.6 #Gaussian downsampling deviation factor

    I3loc_all = []
    headers_all = []
    goodlines_all = []
    badlines_all = []

    #Select data from folder - loopable 
    for satidx, satname in enumerate(satlist):
        for frameidx, framerange in enumerate(framerangelist):
            imgname=imgnamelist[frameidx]
            prename='%s_%s'%(imgname,satname)
            linename=linenamelist[frameidx]
            if linename=='NULL':
                tempname=prename
            else:
                tempname='%s_%s'%(linename,satname)
            if os.path.exists('%s/record_%s.json'%(outfolder,tempname)) and cluster==2 and makeimg==0:
                print('record found for %s'%tempname)
                continue

            input_dir = '%s/%s'%(satfolder,satname)
            print('imporing data for %s '%satname)

            I3loc_all.append(input_dir)
            if datatype=='bin':
                I3, headers = import_bin(input_dir, savedata=0, subsample=1,
                    framerange=framerange, scale=imscale,median=median,shift=shift)
            elif datatype=='fits':
                printheader=0
                I3, headers = import_fits(input_dir, savedata=0, subsample=1,
                    framerange=framerange, scale=imscale, printheader=printheader,gauss=[scale,sigma],median=median,shift=shift,sig=sig,binfactor=binfactor)
            I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))
            #frameidx.extend([framerange[0]:framerange[1]])
            #volume=volume.vstack
            if linename=='NULL':
                tempname=prename
            else:
                tempname='%s_%s'%(linename,satname)
            headers_all.extend(headers)

            print('imporing lines for %s '%tempname)
            clustertime=time.time()
            linesexist=os.path.exists('%s/goodlines_%s.npy'%(outfolder,tempname))
            print('%s/goodlines_%s.npy'%(outfolder,tempname))
            print('exists? ',os.path.exists('%s/goodlines_%s.npy'%(outfolder,tempname)))
            print('resolve? ',resolve==0)

            if not linesexist:
                print('ERROR: associate_all() requires clustering to have been completed on a windowed set.  Please run main.')
                quit()

            goodlines = np.load("%s/goodlines_%s.npy"%(outfolder,tempname))
            badlines = np.load("%s/badlines_%s.npy"%(outfolder,tempname))

            #CORRECT FOR FRAME OFFSETS
            goodlines[:,[2,5]] = goodlines[:,[2,5]] + framerange[0]
            badlines[:,[2,5]] = badlines[:,[2,5]] + framerange[0]
            #Add to list
            print('LOCAL LINES')
            print('good/bad = ',[len(goodlines),len(badlines)])
            goodlines_all.append(goodlines)
            badlines_all.append(badlines)

        #arrange concatenation
        #goodlines_all =sum(goodlines for goodlines in goodlines_all)
        goodlines = np.vstack(goodlines_all)
        badlines = np.vstack(badlines_all)
        print('GLOBAL LINES')
        print('good/bad = ',[len(goodlines),len(badlines)])
        #headers_all = sum(headers for headers in headers_all)
        #define names
        prename='%s_%s'%(allname,satname)
        if linename=='NULL':
                tempname=prename
        else:
                tempname='%s_%s'%(alllinename,satname)

        fullshape = I3.shape
        print('FRAME SHAPE: ',fullshape)
        aa,bb,cc = fullshape
        cc2 = len(headers_all)
        fullshape = [aa,bb,cc2]
        print('NEW SHAPE: ',fullshape)

        #Run associate
        clustertime=time.time()
        linesexist=os.path.exists('%s/goodlines_%s.npy'%(outfolder,tempname))
        print('DISABEL CLUSTER ASSOC, use save')
        if cluster or (not linesexist):
            #goodlines, badlines = detect_outliers(fullshape,goodlines_all,
            #                    folder=outfolder, savename=tempname)
            print('associating...')
            if runoutlier==1:
                starfilter=4 #prepare ROUGH trajectories??
                densityfilter=0
            else:
                starfilter=0#4
                densityfilter=0

            #non-vertical association prior to outlier extraction
            #(only need star tracks, too much noise)
            starfilter=4
            densityfilter=0
            goodlines_all = np.copy(goodlines)
            _, goodlines, rejlines, _ = associate_lines(np.arange(len(goodlines_all)), goodlines_all,len(goodlines_all), fullshape,
                                            postcluster=2, starfilter=starfilter, densityfilter=densityfilter, folder=outfolder, name=tempname,spreadcorr=1)
            #badlines = badlines_all
            print('input ',len(goodlines_all))
            print('goodlines ',len(goodlines))
            print('rejlines  ',len(rejlines))
            print('badlines  ',len(badlines))
            #GOODLINES ARE ASSOCIATED, NOT BAD
            goodlines = np.vstack([goodlines,rejlines])
            #badlines = np.vstack([badlines,rejlines])


            if runoutlier==1:
                print('RUNNING OUTLIER DETECTION ON FULL SET')
                #lines = np.vstack([goodlines,badlines])
                #goodlines, badlines = detect_outliers_new(fullshape,lines,folder=outfolder,
                #    savename=tempname,e2=0,injectScale=-1,subprocess_count=20,
                #    postcluster=1,runoutlier=1,spreadcorr=len(linenamelist))
                #'''

                #do i want to feed outlier EVERYTHING or just reflines?  JUST REJLINES
                #lines = np.vstack([goodlines,rejlines])
                lines = np.copy(goodlines) #includes associated and unassociate "non-clustered" data

                k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
                el = np.arccos((lines[:,5]-lines[:,2])/k_len)
                az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))


                #k_len = k_len
                linesNorm = np.array([k_len*np.cos(az)*np.sin(el),k_len,lines[:,6],
                            k_len*np.sin(az)*np.sin(el),k_len*np.cos(el),lines[:,-1]]).T
                print('inputing %d lines'%len(linesNorm))
                #CLUSTERING MAY KILL OFF EVERYTHING, GIVE GOODLINES AND FIND ONLY NEW BAD THINGS
                goodidxfull,badidxfull = outlier_alg(linesNorm, 3, 3, 0, runoutlier,name=tempname,folder=imgfolder)
                #goodlines=lines[goodidxfull]
                idxlist = np.zeros((len(lines)))
                idxlist[goodidxfull]=1
                goodlines_cluster = lines[idxlist==1]
                badlines_cluster = lines[idxlist==0]
                #'''
                goodlines = goodlines_cluster
                print('good_cluster ',len(goodlines))
                print('bad_cluster ',len(badlines_cluster))
                badlines = np.vstack([badlines,badlines_cluster])
                print('badlines  ',len(badlines))

                lines = np.copy(goodlines) #includes associated and unassociate "non-clustered" data

                k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
                el = np.arccos((lines[:,5]-lines[:,2])/k_len)
                az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
                #k_len = 1.
                linesNorm = np.array([k_len*np.cos(az)*np.sin(el),k_len,lines[:,6],
                            k_len*np.sin(az)*np.sin(el),k_len*np.cos(el),lines[:,-1]]).T

                #badlines_in = np.copy(badlines)
                print('RUNNING SECOND LAYER OF OUTLIER DETECTION, IF POSSIBLE')
                print('ATTEMPTING 3D PCA to a 3D PROBLEM ...',flush=True)
                print('inputing %d lines'%len(linesNorm))
                #try:
                goodidxfull,badidxfull = outlier_alg(linesNorm, 3, 3, 0, runoutlier,name='%sPass1'%tempname,folder=imgfolder)
                #except Exception as e:
                #    print(e)
                #    print('passing all lines as contingency...')
                #    goodidxfull = np.ones((len(linesNorm),))
                #    badidxfull = np.zeros((len(linesNorm),))
                #goodlines=lines[goodidxfull]
                idxlist = np.zeros((len(lines)))
                idxlist[goodidxfull]=1
                goodlines_cluster = lines[idxlist==1]
                badlines_cluster = lines[idxlist==0]
                goodlines = goodlines_cluster
                print('good_cluster ',len(goodlines))
                print('bad_cluster ',len(badlines_cluster))
                #badlines = np.vstack([badlines_in,badlines_cluster])
                badlines = np.vstack([badlines,badlines_cluster])
                print('badlines  ',len(badlines))
                print('ATTEMPTING 3D PCA to a 1D PROBLEM ...',flush=True)
                print('inputing %d lines'%len(linesNorm))
                #try:
                goodidxfull,badidxfull = outlier_alg(linesNorm, 3, 1, 0, runoutlier,name='%sPass2'%tempname,folder=imgfolder)
                #except Exception as e:
                #    print(e)
                #    print('passing all lines as contingency...')
                #    goodidxfull = np.ones((len(linesNorm),))
                #    badidxfull = np.zeros((len(linesNorm),))
                #goodlines=lines[goodidxfull]
                idxlist = np.zeros((len(lines)))
                idxlist[goodidxfull]=1
                goodlines_cluster = lines[idxlist==1]
                badlines_cluster = lines[idxlist==0]
                goodlines = goodlines_cluster
                print('good_cluster ',len(goodlines))
                print('bad_cluster ',len(badlines_cluster))
                #badlines = np.vstack([badlines_in,badlines_cluster])
                badlines = np.vstack([badlines,badlines_cluster])
                print('badlines  ',len(badlines))

            #WITH A RECDUCED SET, NOW ENABLE VERTICLE ASSOCIATION
            starfilter=0
            densityfilter=0
            #extract meaningful lines from the cluster_good set (rejecing unclusterd)
            #NOW redefine goodlines_all input as the good_cluster set!
            goodlines_all = np.copy(goodlines)
            _, goodlines, rejlines, _ = associate_lines(np.arange(len(goodlines_all)), goodlines_all,len(goodlines_all), fullshape,
                                            postcluster=2, starfilter=starfilter, densityfilter=densityfilter, folder=outfolder, name=tempname,spreadcorr=1)
            #badlines = badlines_all
            print('input ',len(goodlines_all))
            print('goodlines ',len(goodlines))
            print('rejlines  ',len(rejlines))
            print('badlines  ',len(badlines))
            #failed association is still a valid track, just unassociated
            goodlines = np.vstack([goodlines,rejlines])
            #Left with goodlines assoc and badlines
            #badlines = np.vstack([badlines,rejlines])


            print("saving to... %s/goodlines_%s.npy"%(outfolder,tempname))
            np.save("%s/goodlines_%s.npy"%(outfolder,tempname),goodlines)
            np.save("%s/badlines_%s.npy"%(outfolder,tempname),badlines)
        else:
            goodlines = np.load("%s/goodlines_%s.npy"%(outfolder,tempname))
            badlines = np.load("%s/badlines_%s.npy"%(outfolder,tempname))
        clustertime=time.time()-clustertime
        print('CLUSTER RUNTIME: %.2f sec (%.2f min)\n\n'%(clustertime,clustertime/60.))

        if not makeimg==0:
            if not printcluster==0:
                print('WARNING: print clustering not enabled.  setting printcluster=0')
                printcluster=0

            if printcluster==0:
                print_detections_window(np.copy(I3),goodlines,badlines,folder=imgfolder,savename=tempname,
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

        astrotime=time.time()
        jsonexists = os.path.exists('%s/record_%s.json'%(outfolder,tempname))
        if solveastro==1 or (solveastro==2 and jsonexists==0):
            print('creating kernel:')
            print('starting astr0...')
            makejson=1
            aa,bb,cc = I3.shape
            headersnew = run_astrometry(I3,goodlines, badlines, headers_all, scale=(aa,bb),
                folder=outfolder,savename=tempname,makejson=makejson,tle=tle,
                binfactor=binfactor,imgastro=imgastro,subprocess_count=subprocess_count,I3loc=I3loc_all,frames=framerangelist,imscale=imscale,median=median,shift=shift,datatype=datatype)
        astrotime=time.time()-astrotime
        print('ASTRO RUNTIME: %.2f sec (%.2f min)\n\n'%(astrotime,astrotime/60.))

        '''
        ## RUN ASTROMETRY AND UPDATE HEADERS
        aa,bb,cc = I3.shape
        headersnew = run_astrometry(goodlines, badlines, headers, scale=(aa,bb), folder=outfolder,savename=prename)
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

