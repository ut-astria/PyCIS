'''
PyCIS - Python Computational Inference from Structure

pylib/detect_outliers.py: 2nd-order classification by clustering and outlier rejection
    naive partitioning method

TODO:
  Update for track association using 3rd layer of point alignment detection

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

import os
import argparse
import numpy as np
import scipy as sp
from scipy import stats 
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Numpy Settings
np.set_printoptions(precision=2)
np.seterr(all='ignore')
import time
from datetime import datetime
import pandas as pd
from multiprocessing import Pool, cpu_count,get_context,set_start_method
from itertools import chain
import cv2


def RunBayes(k,xnum, X, wcp=1e-2,wtype='distribution',mpp=1e-2,cp=1):
    '''
    Compute GMM to rank r<k using variational bayesian inference
    Only used for single-k fitting in current method but preserved 
    for consistency with legacy system
    '''
    k=int(k)
    init_params="kmeans"#random"#"kmeans"#"random"
    model = BayesianGaussianMixture(n_components=k,covariance_type='full', 
                weight_concentration_prior=wcp,
                weight_concentration_prior_type='dirichlet_%s'%wtype, 
                mean_precision_prior=mpp, 
                covariance_prior=cp*np.eye(xnum),
                init_params=init_params, max_iter=1000,random_state=2)
    model.fit(X)
    return model

def PCA(linesNormIn, numvars=6):
    '''
    Compute PCA
    '''
    # linesNorm: 
    #    [0=length, 1=maxwidth, 2=minwidth, 3=az, 4=el, 5=nfa]
    linesNorm = np.copy(linesNormIn)

    #select features for PCA
    if numvars==3:
        #width-agnostic features (for joined lines with reliable lengths)
        #   (length, az, el)
        linesNorm=linesNorm[:,[0,3,4]]
    elif numvars==1:
        # length-agnostic features (in case of fragmentation)
        #   (maxwidth, minwidth, az, el)
        linesNorm=linesNorm[:,[3,3]]
    elif numvars==2:
        # length-agnostic features (in case of fragmentation)
        #   (maxwidth, minwidth, az, el)
        linesNorm=linesNorm[:,[3,4]]
    elif numvars==4:
        # length-agnostic features (in case of fragmentation)
        #   (maxwidth, minwidth, az, el)
        #linesNorm=linesNorm[:,[1,2,3,4]]
        linesNorm=linesNorm[:,[0,1,3,4]] #maxwid
        #linesNorm=linesNorm[:,[0,2,3,4]] #minwid
    else:
        #Simply select data
        linesNorm = linesNorm[:,:(numvars)]
        print('dim: ',linesNorm.shape[-1])

    for x in range(linesNorm.shape[1]):
        if np.all(linesNorm[:,x]==linesNorm[0,x]):
            print("ERROR: detect_outliers: all elements on linesNorm index %d are identical"%x,flush=True)
            print("\t this may be due to, say, only detecting spatial features with elevation pi/2",flush=True)
            quit()

    ## WHITENING
    #standardize data to zero mean unit variance
    linesNorm = standardize(linesNorm,0)
    #determine eigenbasis
    covar = np.cov(linesNorm.T)
    evals,evecs = np.linalg.eig(covar)
    esort = np.argsort(-evals)
    evals = evals[esort]
    evecs = evecs[:,esort]
    #project data into eigenspace to uncorrelate data
    proj = evecs[:,:]
    linesNormPCA = linesNorm.dot(proj)

    #return selected features, projected features, and projection
    return linesNorm, linesNormPCA, proj

def standardize(XIn, method):
    '''
    Perform whitening or unit scaling 
    '''    
    XFull=np.copy(XIn)
    Y=np.copy(XFull)
    for column in range(XFull.shape[1]):
        tempArray = np.empty(0)

        if method==0: #Whitening
            mean = np.mean(Y[:,column])
            std = np.std(Y[:,column])
            for element in XFull[:,column]:
                tempArray = np.append(tempArray, ((element - mean) / std))
        else: #Unit Scaling
            amax = np.amax(Y[:,column])
            amin = np.amin(Y[:,column])
            for element in XFull[:,column]:
                tempArray = np.append(tempArray, ((element - amin) / (amax-amin)))

        XFull[:,column] = tempArray
    return XFull

def run_pca(linesFilt,linesNorm,pcanum,xnum): 
    '''
    Perform the whitening and scaling processes checking sizing and low-rank projection
    '''
    #Compute PCA on select features of training data 
    _, linesFiltPCA,proj = PCA(linesFilt, pcanum)
    # Filter testing set to features of choice
    linesNormFilt, _,_ =   PCA(np.copy(linesNorm), pcanum)
    #Project testing data to principal components of training data 
    linesFullPCA = np.copy(linesNormFilt).dot(proj)
    
    #Define training (filt) and testing (full) data, 
    # as results of PCA analysis to rank r=k-1 PCA projection
    XFilt=linesFiltPCA[:,:xnum]
    XFull=linesFullPCA[:,:xnum]
    if XFull.ndim==1:
        XFull = XFull[:,np.newaxis]
        XFilt = XFull[:,np.newaxis]
    
    # Scale data to the [0,1]**k unit cube 
    XFull = standardize(XFull, 1)
    XFilt = standardize(XFilt, 1)
    return XFilt,XFull



def format_lines(lines,a=10,aa=4906, filt=0):
    '''
    Format lines for matrix processing
    '''
    #Convert line vector to feature matrix 
    l = np.size(np.asarray(lines))/a
    lines = np.reshape(np.asarray(lines).T,(-1,a),order='F')

    #Ensure NFA is never infinite
    nanfilt2 = ~np.isfinite(lines[:,-1])
    print('Setting %d/%d lines with infinite NFA to have 1e6 NFA'%(np.count_nonzero(nanfilt2),len(lines)))
    lines[nanfilt2,-1]=1e6

    #Remove lines with infinite elements (should not remove anything...)
    nanfilt1 = np.isfinite(lines).all(axis=1)
    print('Removing %d/%d lines due to infinite elements'%(np.count_nonzero(~nanfilt1),len(lines)))
    lines = lines[nanfilt1]

    #Remove lines of invalid length   
    k_len=np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
    sortklen= k_len<=0
    print('%d/%d removed for zero-length'%(np.count_nonzero(sortklen),len(sortklen)))
    lines=lines[~sortklen]
    k_len=np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
    sortklen =  ~np.isfinite(k_len)
    print('%d/%d removed for inf-length'%(np.count_nonzero(sortklen),len(sortklen)))
    
    lines=lines[~sortklen]
    #(Optionally) remove edge lines
    w=0#.1
    if w!=0:
        print('Removing edge lines to prevent gradient-computation anomaly effects on solution ')
        lines = lines[lines[:,0]>w*aa]
        lines = lines[lines[:,0]<(1-w)*aa]
        lines = lines[lines[:,1]>w*aa]
        lines = lines[lines[:,1]<(1-w)*aa]
        lines = lines[lines[:,3]>w*aa]
        lines = lines[lines[:,3]<(1-w)*aa]
        lines = lines[lines[:,4]>w*aa]
        lines = lines[lines[:,4]<(1-w)*aa]
        #If filtering turned on (off by default) remove small lines or near-horizontal lines 
        #  eg. noise and star streaks - 'naive solution' 
        if filt==1:
            k_len=np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
            tol = 1*np.pi/(64)
            lines=lines[(np.pi/2. - np.abs(np.arccos(np.abs(lines[:,5]-lines[:,2])/k_len)))>tol]
            k_len=np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
            zsize = np.sqrt(2)*27
            lines = lines[k_len>zsize]

    return lines

def ball_nfa(k,n,p,Ntests,minpts,maxpts):
    ''' compute nfa of cluster '''
    if k<=minpts:
        nfa=-101.
    elif k>=maxpts:
        nfa=-101.
    else:
        r = float(k)/float(n)
        if (r==1 or r==0):
            nfa=-101.
        else:
            tail = stats.binom.logsf(k=k,n=n,p=p) #use ninit
            nfa = -(tail/np.log(10.))-np.log10(Ntests)
            if nfa>200: #0823 THERES A TYPO IN THE 2008 TEXT, SHOULD BE KL
                tail = -n*( r*np.log(r/p) + (1.-r) * np.log( (1.-r)/(1.-p) )  )
                nfa = -(tail/np.log(10.))-np.log10(Ntests)
            if np.math.isinf(nfa):
                nfa=1001.
            elif np.math.isnan(nfa):
                print('ISNAN')
                quit()
    return nfa     

def acclust(radius,maxradius,dims,Ntests,Nrads,XFullIn):
    '''
    Parallelizable proximity clustering to build NFA lookup table 
    
    Input: 
        radius:    tested radius for clustering
        maxradius: maximum radius of data
        dims:      data dimensionality 
        Ntests:    total hypothetical number of tests
        Nrads:     number of radii in Ntests 
        XFullIn:   data to cluster on 
    
    Output: 
    

    '''

    XFull = np.copy(XFullIn)
    #Number of total points in original set vs those currently measurable
    n = int(Ntests/Nrads)
    nb = len(XFull)
    # Probability, minimum and maximum set sizes
    prob = ((2.*radius) / maxradius )**dims
    p=prob
    minpts = max(int(np.ceil(p*n)+1),2) #uniform case is lower bound
    maxpts = 1.* n 
    #Build tree and storage
    tree = sp.spatial.KDTree(np.copy(XFull))
    maxn = 1000 #maximum size of comparison tree 
    pt1list = []
    balllist = []
    radiuslist = []
    nfalist = []

    #Subsampling case  
    if nb>maxn:
        for i in range(0,nb,maxn):
            if i>=nb:
                continue
            j = min(nb,i+maxn)
            XFullB = np.copy(XFull[i:j,:])
            tree2 = sp.spatial.KDTree(XFullB)
            ball = tree2.query_ball_tree(tree,r=radius, p=1)
            for pt1 in range(len(ball)):
                ptball =  ball[pt1]
                k = len(ptball)
                nfa = ball_nfa(k,n,p,Ntests,minpts,maxpts)
                pt1list.append(pt1)
                balllist.append(ball[pt1])
                radiuslist.append(radius)
                nfalist.append(nfa)
    else:
        tree2 = sp.spatial.KDTree(XFull)
        ball = tree2.query_ball_tree(tree,r=radius, p=1)
        for pt1 in range(len(ball)):
            ptball =  ball[pt1]
            k = len(ptball)
            #threshold = min k s.t. NFA(k)=Nt * p^k <= eps
            #this is a loose threshold
            threshold = -np.log10(Ntests)/np.log10(p)
            nfa = ball_nfa(k,n,p,Ntests,minpts,maxpts)
            pt1list.append(pt1)
            balllist.append(ball[pt1])
            radiuslist.append(radius)
            nfalist.append(nfa)

    return [pt1list,balllist,radiuslist,nfalist]


def detect_outliers(shape,lines=[],folder='',savename='temp',args=None,stars=0,e2=0,subprocess_count=10):
    '''
    Pipeline for NFA outlier detection using PCA/GMM 
    Input: 
        Image: 3D data cube of image data
        Lines: [x1,y1,z1,x2,y2,z2,width1,width2,precision,centerline nfa]
    Output:
        Goodlines.npy: 2nd order meaninful lines [x1,y1,z1,x2,y2,z2,width1,width2,precision,centerline nfa]
        Badlines.npy:  1st order meaninful lines [x1,y1,z1,x2,y2,z2,width1,width2,precision,centerline nfa]
    '''


    '''
    ------------------------------------------------------
    '''
    print('Filter Image and Line Data ')
    try:
        aa,bb,cc = shape
    except:
        print('ERROR: EMPTY IMG')
        quit()
    print('Filter Image and Line Data ')
    '''
    # Format image data 
    if len(img)!=0:
        v = np.array(np.stack((img,)*3,axis=-1))
        aa,bb,cc,_ = v.shape
    else:
        print('ERROR: EMPTY IMG')
        quit()
    '''
    #Format lines into proper matrix shape, and 
    lines = format_lines(lines,aa=aa,filt=0)

    linessave=np.copy(lines)
    #Outlier detection only viable if there exist a 
    # statistically meaninful number of events 
    if 1==1:

        '''
        ------------------------------------------------------
        '''
        print('Construct Feature Vectors From Line Data')
        #   linkes[k,:] = [x,y,z,x,y,z,w,w,ptol,nfa]
        #   linesNorm: length, maxwidth, minwidth, az[-pi,pi], el[0,pi/2], nfa

        # Store length
        k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
        # Store NFA value from LSD
        nfa = lines[:,-1]
        # Store angular tolerance from LSD
        ptol = lines[:,-2]

        # Sort widths so that w1_sort contains the maximum width
        w1 = lines[:,-4]
        w2 = lines[:,-3]
        w1_sort = np.copy(w1)
        w2_sort = np.copy(w2)
        for k in range(len(w1)):
            w1_sort[k] = min(w1[k],w2[k])
            w2_sort[k] = max(w1[k],w2[k])

        # Compute angle from endpoint sets 
        idx = lines[:,5]<lines[:,2]
        temp = np.copy(lines[idx,:])
        lines[idx,:3]=temp[:,3:6]
        lines[idx,3:6]=temp[:,:3]

        el = np.arccos((lines[:,5]-lines[:,2])/k_len)
        az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
        

        #Define the data to be used
        #   linkes[k,:] = [x,y,z,x,y,z,w,w,ptol,nfa]
        #   linesNorm: length, maxwidth, minwidth, az[-pi,pi], el[0,pi/2], nfa
        #linesNorm = np.array([k_len,w1_sort,w2_sort,az,el,nfa]).T
        linesNorm = np.array([k_len*np.cos(az)*np.sin(el),k_len,w2_sort,
                    k_len*np.sin(az)*np.sin(el),k_len*np.cos(el),nfa]).T

        '''
        ------------------------------------------------------
        '''
        print('Set some variables')
        #Number of PCA components for decomposition
        #unit direction + length on separate axis
        pcanum=3#length, az, el (width and nfa agnostic)
        #Number of PCA components for GMM separation
        xnum = 3#-2 #max(pcanum-2,2)

        '''
        ------------------------------------------------------
        '''
        print('Separate Training/Testing Data')

        linesFilt = np.copy(linesNorm)
        XFilt,XFull = run_pca(linesFilt,np.copy(linesNorm),pcanum,xnum)        

        '''
        ------------------------------------------------------
        '''
        print('Proximity Clustering ')
        '''
        ------------------------------------------------------
        '''

        #Number of clusters to test
        maxradius = 1.
        dims = XFull.shape[1]
        Nrads = 100;
        Ntests = Nrads * len(XFull);
        maxfact = .5*(.54**(1./dims))
        radiusset = np.linspace(0,maxradius*maxfact,Nrads+1) #should not exceep .25
        radiusset = np.geomspace(radiusset[1],maxradius*maxfact,num=Nrads+1) #p < .25
        radiusset = radiusset[radiusset>0]  
        radiusset = radiusset[:-1]
        print('radiusset length:',len(radiusset))
        Ntests = np.copy(len(radiusset) * len(XFull));
        print('dims %d, NTests %.2f, lognfa %.2f'%(dims,Ntests,-np.log10(Ntests))) 
        print('Computing...')
        starttime=datetime.now()
         
        #parallelization info
        numsteps= subprocess_count
        process_count = min(numsteps,cpu_count())
        chunks = int(len(radiusset)/process_count)
        runacclust=1;

        #storage for clustering groups
        goodidxfull = np.asarray(range(len(XFull)))
        CLUST1idxfull = np.empty((0))
        CLUST2idxfull = np.empty((0))
        CLUST3idxfull = np.empty((0))
        CLUSTXidxfull = np.empty((0))
        REMAINidxfull = np.empty((0))
        OUTLIERidxfull = np.empty((0))
        badidxfull = np.asarray(range(len(XFull)))
        XNocluster = np.copy(XFull)
        XAllClusters = []


        templinesFilt = np.copy(linesNorm)
        counter = 0;
        while runacclust==1:
            counter+=1;

            #Reproject and cluster data
            _,XNocluster = run_pca(templinesFilt,templinesFilt,pcanum,xnum)        
            iterable=[]
            print('XNocluster len: ',len(XNocluster))
            for radius in radiusset:
                iterable.append([radius,maxradius,dims,Ntests,len(radiusset),XNocluster])
            with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
                results=pool.starmap(acclust,iterable,chunksize=chunks)
            pt1list = []
            balllist = []
            radiuslist = []
            nfalist = []
            for r in results:
                pt1list.extend(r[0])
                balllist.extend(r[1])
                radiuslist.extend(r[2])
                nfalist.extend(r[3])
            store = pd.DataFrame(
                {'core': pt1list, 'cluster': balllist, 
                'radius': radiuslist ,   'nfa': nfalist},
                columns=['core','cluster','radius','nfa'])

            #print('RUNTIME: ',datetime.now()-starttime)
            
            #Identify maximally meaningfu lcluster
            newnfalist = np.amax(store['nfa'])
            if not np.isscalar(newnfalist):
                newnfalist = newnfalist[0]
            print('max nlog10nfa',newnfalist)
            if newnfalist>e2:
                #get list of items at maximum NFA
                nfastore = store.loc[store['nfa']==newnfalist]
                clustmask = np.zeros((len(XNocluster)))
                #print('cluster size A: ', len(nfastore['cluster']))
                subnfaidx= 0 ; clustsize=0;#len(XNocluster);
                tempnfastore = nfastore['cluster'].tolist()
                if len(nfastore)>1:
                    #remove the largest set with maximum nfa
                    #if every point refers to the same cluster, all are removed
                    for ccc,clustidx in enumerate(tempnfastore):
                        if len(clustidx)>clustsize:
                            clustsize=len(clustidx)
                            subnfaidxidx = ccc
                    for clustidx in tempnfastore[subnfaidx]:
                        clustmask[clustidx] = 1
                else:
                    for clustidx in tempnfastore:
                        clustmask[clustidx] = 1
                #print('cluster size B: ', np.count_nonzero(clustmask))

                #Assign clusters to storage
                clustmask = clustmask==1
                XCluster = XNocluster[clustmask,:]
                XAllClusters.append(XCluster)
                if counter==1:
                    CLUST1idxfull=np.append(CLUST1idxfull,goodidxfull[clustmask])
                elif counter==2:
                    CLUST2idxfull=np.append(CLUST2idxfull,goodidxfull[clustmask])
                elif counter==3: 
                    CLUST3idxfull=np.append(CLUST3idxfull,goodidxfull[clustmask])
                else: 
                    CLUSTXidxfull=np.append(CLUSTXidxfull,goodidxfull[clustmask])
                XNocluster = XNocluster[~clustmask,:]
                goodidxfull = goodidxfull[~clustmask]
                templinesFilt = templinesFilt[~clustmask,:]
            else:
                runacclust=0
            if len(XNocluster)==0:
                runacclust=0
    

    #Get gaussian fit of data    
    _,XNocluster = run_pca(templinesFilt,templinesFilt,pcanum,xnum)        
    XFilt =np.copy(XNocluster)
    #just fit one gaussian
    model = RunBayes(1,np.copy(xnum),np.copy(XFilt),wtype='process')
    X = np.copy(XNocluster)
    predict_proba = model.predict_proba(X) 
    score_samples = -model.score_samples(X) / np.log(10)
    #select rank r<k data 
    predict_proba2 = predict_proba.squeeze()#2[:,weightsort]
    weights2 = model.weights_.squeeze()#[weightsort]
    means2 = model.means_.squeeze()#[weightsort]
    covar2 = model.covariances_.squeeze()#[weightsort]
    #precompute some constants to speed up further iteration  
    if covar2.ndim>1:
        covarinv = np.linalg.inv(covar2)
    else:
        covarinv = 1./covar2
        means2 = means2.reshape((1,1))
        covarinv = covarinv.reshape((1,1))
    #initialize memory 
    mahan = np.copy(score_samples)

    fn,fm = X.shape
    cc = 1.
    crv = stats.chi2(fm)#,scale=1.)
    #for each line sample...
    method = 'max'
    for i in range(len(score_samples)):
        mahanrow = ((X[i]-means2).T @  covarinv @ (X[i]-means2))
        mahanrow = cc * mahanrow 
        mahan[i] = 1.-crv.cdf(mahanrow) 

    score_samples = -np.log10(mahan) - np.log10(len(mahan))

    
    eps=e2#-np.inf #-log10(nfa)<-log10(e=1)
    #print('DISABLING OUTLIER')
    #e2=-np.inf
    print('min: %.2f, max: %.2f, eps: %.2f'%(min(score_samples),max(score_samples),eps))
    
    #Compare NFA to threshold 
    tempgoodidxfull = np.copy(np.asarray(goodidxfull))
    goodidxfull = tempgoodidxfull[score_samples>e2]
    badidxfull = tempgoodidxfull[score_samples<=e2]

    print('Xshape,',X.shape)
    print('scireshape,',score_samples.shape)

    saveoutlier  = np.copy(X[score_samples>e2])
    saveremain = np.copy(X[score_samples<=e2])

    tempgoodidxfull = np.copy(np.asarray(goodidxfull))
    goodidxfull = np.zeros((len(XFull)))
    CLUST1idxfull2 = np.zeros((len(XFull)))
    CLUST2idxfull2 = np.zeros((len(XFull)))
    CLUST3idxfull2 = np.zeros((len(XFull)))
    CLUSTXidxfull2 = np.zeros((len(XFull)))
    REMAINidxfull2 = np.zeros((len(XFull)))
    OUTLIERidxfull2 = np.zeros((len(XFull)))

    CLUST1idxfull = np.asarray(CLUST1idxfull.flatten(),dtype=int)
    CLUST2idxfull = np.asarray(CLUST2idxfull.flatten(),dtype=int)
    CLUST3idxfull = np.asarray(CLUST3idxfull.flatten(),dtype=int)
    CLUSTXidxfull = np.asarray(CLUSTXidxfull.flatten(),dtype=int)

    goodidxfull[tempgoodidxfull]=1
    CLUST1idxfull2[CLUST1idxfull]=1
    CLUST2idxfull2[CLUST2idxfull]=1
    CLUST3idxfull2[CLUST3idxfull]=1
    CLUSTXidxfull2[CLUSTXidxfull]=1
    REMAINidxfull2[badidxfull]=1
    OUTLIERidxfull2[tempgoodidxfull]=1

    goodidxfull = goodidxfull==1
    CLUST1idxfull = CLUST1idxfull2==1
    CLUST2idxfull = CLUST2idxfull2==1
    CLUST3idxfull = CLUST3idxfull2==1
    CLUSTXidxfull = CLUSTXidxfull2==1
    REMAINidxfull = REMAINidxfull2==1
    OUTLIERidxfull = OUTLIERidxfull2==1
    #tempgbadidxfull = np.copy(np.asarray(badidxfull))
    badidxfull = ~goodidxfull
    
    lines_full = np.copy(linessave)
    goodlinesfull = lines_full[goodidxfull]
    badlinesfull = lines_full[badidxfull]
   
    print('Total lines: %d, Accepted: %d, Rejected: %d'%(len(lines_full), 
        np.count_nonzero(goodidxfull),np.count_nonzero(badidxfull) ))
    '''
    print('Accepted:')
    print(goodlinesfull)
    print('Rejected:')
    print(badlinesfull[:10,:] )
    Xgood= XFull[goodidxfull,:]
    Xbad = XFull[badidxfull,:]
    print('Accepted:')
    print(Xgood)
    print('Rejected:')
    print(Xbad[:10,:] )
    '''
    np.save('%s/goodlines_%s.npy'%(folder,savename),goodlinesfull)
    np.save('%s/badlines_%s.npy'%(folder,savename),badlinesfull)
    
    np.save('%s/CLUST1_%s.npy'%(folder,savename),lines_full[CLUST1idxfull])
    np.save('%s/CLUST2_%s.npy'%(folder,savename),lines_full[CLUST2idxfull])
    np.save('%s/CLUST3_%s.npy'%(folder,savename),lines_full[CLUST3idxfull])
    np.save('%s/CLUSTX_%s.npy'%(folder,savename),lines_full[CLUSTXidxfull])
    np.save('%s/REMAIN_%s.npy'%(folder,savename),lines_full[REMAINidxfull])
    np.save('%s/OUTLIER_%s.npy'%(folder,savename),lines_full[OUTLIERidxfull])
    


    ''' 
    import cv2 
    viridis_seed = np.array([np.linspace(1,255,num=5)]).astype(np.uint8)
    viridis_list = cv2.applyColorMap(viridis_seed,cv2.COLORMAP_VIRIDIS)[0]
    #print('list0')
    #print(viridis_list)
    viridis_list = np.asarray(viridis_list)[:,::-1]/255
    #print('list1')
    #print(viridis_list)

    
    plt.figure()
    plt.subplot(1,2,1) #clusters + remainders
    templinesFilt = np.copy(linesNorm)
    _,X = run_pca(templinesFilt,templinesFilt,pcanum,xnum)        
    vl = viridis_list[0]
    plt.scatter(x=X[REMAINidxfull,0],y=X[REMAINidxfull,1],color=(vl[0],vl[1],vl[2]))
    plt.scatter(x=X[OUTLIERidxfull,0],y=X[OUTLIERidxfull,1],color=(vl[0],vl[1],vl[2]))
    vl = viridis_list[1]
    plt.scatter(x=X[CLUSTXidxfull,0],y=X[CLUSTXidxfull,1],color=(vl[0],vl[1],vl[2]))
    vl = viridis_list[2]
    plt.scatter(x=X[CLUST3idxfull,0],y=X[CLUST3idxfull,1],color=(vl[0],vl[1],vl[2]))
    vl = viridis_list[3]
    plt.scatter(x=X[CLUST2idxfull,0],y=X[CLUST2idxfull,1],color=(vl[0],vl[1],vl[2]))
    vl = viridis_list[4]
    plt.scatter(x=X[CLUST1idxfull,0],y=X[CLUST1idxfull,1],color=(vl[0],vl[1],vl[2]))
    plt.title("Meaningful Clusters \n 1st projection")
    plt.subplot(1,2,2) #remainders + outliers
    plt.scatter(x=saveremain[:,0],y=saveremain[:,1],color=(vl[0],vl[1],vl[2]))
    plt.scatter(x=saveoutlier[:,0],y=saveoutlier[:,1],c='r')
    plt.title("Meaningful Outliers")
    plt.savefig("TESTCLUSTER.png")
    plt.close()     
    '''
    '''
    fig=plt.figure()
    ax = fig.add_subplot(1,2,1,projection='3d') #clusters + remainders
    templinesFilt = np.copy(linesNorm)
    _,X = run_pca(templinesFilt,templinesFilt,pcanum,xnum)        
    vl = viridis_list[0]
    plt.scatter(X[REMAINidxfull,0],X[REMAINidxfull,1],X[REMAINidxfull,2],color=(vl[0],vl[1],vl[2]))
    plt.scatter(X[OUTLIERidxfull,0],X[OUTLIERidxfull,1],X[OUTLIERidxfull,2],color=(vl[0],vl[1],vl[2]))
    vl = viridis_list[1]
    plt.scatter(X[CLUSTXidxfull,0],X[CLUSTXidxfull,1],X[CLUSTXidxfull,2],color=(vl[0],vl[1],vl[2]))
    vl = viridis_list[2]
    plt.scatter(X[CLUST3idxfull,0],X[CLUST3idxfull,1],X[CLUST3idxfull,2],color=(vl[0],vl[1],vl[2]))
    vl = viridis_list[3]
    plt.scatter(X[CLUST2idxfull,0],X[CLUST2idxfull,1],X[CLUST2idxfull,2],color=(vl[0],vl[1],vl[2]))
    vl = viridis_list[4]
    plt.scatter(X[CLUST1idxfull,0],X[CLUST1idxfull,1],X[CLUST1idxfull,2],color=(vl[0],vl[1],vl[2]))
    ax.set_title("Meaningful Clusters \n 1st projection")
    ax=fig.add_subplot(1,2,2,projection='3d') #remainders + outliers
    plt.scatter(saveremain[:,0],saveremain[:,1],saveremain[:,2],color=(vl[0],vl[1],vl[2]))
    plt.scatter(saveoutlier[:,0],saveoutlier[:,1],saveoutlier[:,2],c='r')
    ax.set_title("Meaningful Outliers")
    plt.savefig("TESTCLUSTER3D.png")
    plt.close()     
    '''



    print('Done!\n\n')
    return goodlinesfull, badlinesfull

