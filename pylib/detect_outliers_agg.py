'''
PyCIS - Python Computational Inference from Structure

pylib/detect_outliers.py: 2nd-order classification by clustering and outlier rejection
    agglomerative heirarchical clustering method


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

import os
import time
from datetime import datetime
from itertools import chain
import numpy as np
np.seterr(all='ignore')
np.set_printoptions(precision=2)
import scipy as sp
from scipy import stats 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import functools
from multiprocessing import cpu_count, get_context #,Pool,set_start_method
import subprocess
import signal 
#import tempfile
from pylib.print_detections import interp_frame_xy
from pylib.detect_utils import my_filter, remove_matches_block
 
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

def PCA(linesNormIn, numvars=6,scale=False):
    ''' Compute PCA (Scale=True will normalize the lines to only consider pointing data, default false) '''
    # linesNorm: 
    #    [0=length, 1=maxwidth, 2=minwidth, 3=az, 4=el, 5=nfa]
    linesNorm = np.copy(linesNormIn)
    scalevec=np.ones((linesNorm.shape[0],))
    if scale:
        scalevec = np.copy(linesNorm[:,1])

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
        #linesNorm=linesNorm[:,[1,2]]
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
    
    failflag=0
    if scale:
        linesNorm = np.asarray(linesNorm) / np.asarray(scalevec).squeeze().reshape((-1,1)) #only pointing data, not length 
    for x in range(linesNorm.shape[1]):
        if np.all(linesNorm[:,x]==linesNorm[0,x]):
            failflag=1
    if failflag==0:
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
    else:
        proj = np.eye(linesNorm.shape[0])
        proj = proj[:,linesNorm.shape[1]]
    linesNormPCA = linesNorm.dot(proj)
    #return selected features, projected features, and projection
    return linesNorm, linesNormPCA, proj

def standardize(XIn, method):
    '''    Perform whitening or unit scaling     '''    
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
    '''    Perform the whitening and scaling processes checking sizing and low-rank projection    '''
    #Compute PCA on select features of training data 
    scale=False
    if xnum==2 and pcanum==3:
        xnum=3
        scale=True
    if xnum<0 and pcanum==3:
        xnum=-xnum
        scale=True
    elif xnum<0 and pcanum!=3:
        print('xnum<0 and pcanum!=3 invalid combination')
        quit()
    _, linesFiltPCA,proj = PCA(linesFilt, pcanum,scale=scale)
    # Filter testing set to features of choice
    linesNormFilt, _,_ =   PCA(np.copy(linesNorm), pcanum,scale=scale)
    #Project testing data to principal components of training data 
    linesFullPCA = np.copy(linesNormFilt).dot(proj)
    
    #Define training (filt) and testing (full) data, 
    # as results of PCA analysis to rank r=k-1 PCA projection
    if linesFiltPCA.ndim==1:
        linesFiltPCA=linesFiltPCA[:,np.newaxis]
        linesFullPCA=linesFullPCA[:,np.newaxis]
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
    '''    Format lines for matrix processing    '''
    #Convert line vector to feature matrix 
    l = np.size(np.asarray(lines))/a
    lines = np.asarray(lines)
    if lines.ndim!=2:
        lines = np.reshape(np.asarray(lines).T,(-1,a),order='F')
    elif lines.shape[1]!=a:
        lines = np.reshape(np.asarray(lines).T,(-1,a),order='F')

    #Ensure NFA is never infinite
    initcount = len(lines)
    nanfilt2 = ~np.isfinite(lines[:,-1])
    #print('Setting %d/%d lines with infinite NFA to have 1e6 NFA'%(np.count_nonzero(nanfilt2),len(lines)))
    lines[nanfilt2,-1]=1e6

    #Remove lines with infinite elements (should not remove anything...)
    nanfilt1 = np.isfinite(lines).all(axis=1)
    #print('Removing %d/%d lines due to infinite elements'%(np.count_nonzero(~nanfilt1),len(lines)))
    lines = lines[nanfilt1]

    #Remove lines of invalid length   
    k_len=np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
    sortklen= k_len<=0
    #print('%d/%d removed for zero-length'%(np.count_nonzero(sortklen),len(sortklen)))
    lines=lines[~sortklen]
    k_len=np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
    sortklen =  ~np.isfinite(k_len)
    #print('%d/%d removed for inf-length'%(np.count_nonzero(sortklen),len(sortklen)))
    print('%d/%d lines removed by filter'%(np.count_nonzero(sortklen),initcount))
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
    ''' compute nfa of cluster using a binomial law '''
    if k<=minpts:
        nfa=-101.
    elif k>=maxpts:
        nfa=-101.
    else:
        r = float(k)/float(n)
        if (r==1 or r==0):
            nfa=-101.
        else:
            tail = stats.binom.logsf(k=k,n=n,p=p)
            nfa = -(tail/np.log(10.))-np.log10(Ntests)
            if nfa>200:
                tail = -n*( r*np.log(r/p) + (1.-p) * np.log( (1.-r)/(1.-p) )  )
                nfa = -(tail/np.log(10.))-np.log10(Ntests)
            if np.math.isinf(nfa):
                nfa=1001.
            elif np.math.isnan(nfa):
                print('ISNAN')
                quit()
    return nfa     

def makeinit(lines, R,nparts, dims, NUMPOINTS):
    ''' 
    Initialize the hierarchicial agglomerative clustering tree 
    Inputs:
        lines: set of features to consider
        R: The "N_T" number of tests, total hypperrectangles in domain
        nparts: The number of paritions of the data
        dims: The dimensions of the data 
        NUMPOINTS: The total number of line candidates, in case lines has been reduces by recursive filtering
    Output: 
        df_tree: Pandas table with "leaf" and "cluster" columns.
            leaf: (partition index, partition nfa)
            cluster: set of lines in the respective partition
        linecores: The coordinates of the middle of each parition
    '''
    step=1./nparts
    Ntests=R
    n = NUMPOINTS #len(lines)
    p = step**dims #==1/numcubes
    minpts = max(int(np.ceil(p*n)+1),2) #uniform case is lower bound
    maxpts = 1.* n 
    #Ntests is regions while n is pts, should have Ntests<npts
    print('INIT: nparts %d, Ntests %d, n %d, p %.2e, minpts %d, matplts %d'%(nparts,Ntests,n,p,minpts,maxpts))    
    #Define the center of each parititon, the finest area of the agglomerative method
    cores = np.indices(tuple(np.asarray(np.ones(dims)*(nparts),dtype=int)))
    cores = cores.T.reshape(-1,dims)
    cores = (cores+1.)*step - step/2 #normalize to unit cube
    #Parition the line data according to the this partitioning 
    linecores = np.zeros_like(lines)
    linetree = sp.spatial.KDTree(np.copy(lines))
    coretree = sp.spatial.KDTree(np.copy(cores))
    balls = coretree.query_ball_tree(linetree,r=step/2., p=np.inf)
    #Store the partitioned data
    pt1list = []
    balllist = []
    radiuslist = []
    nfalist = []
    ksum=0
    ptcount = 0
    for pt1 in range(len(balls)):
        ptball =  balls[pt1]
        #balls[pt1] = unique LINE indices mapping to each core cluster
        for pt2 in ptball:
            linecores[pt2,:] = cores[pt1,:]
        k = len(ptball)
        if k==0:
            continue 
        ksum=ksum+k
        nfa = ball_nfa(k,n,p,Ntests,minpts,maxpts)
        pt1list.append(ptcount)
        balllist.append(balls[pt1])
        nfalist.append(nfa)
        ptcount=ptcount+1
  
    #build pandas tree to track nfa layers for cluster idx of orig poins
    df_tree = pd.DataFrame(
        {'leaf': list(zip(pt1list,nfalist)), 'cluster': balllist},
        columns=['leaf','cluster']) #append node: zip(layeridx,nfa)i
    nfatest = df_tree['leaf'].to_list()
    nfatest = np.array(nfatest)[:,1]
    return df_tree,linecores 



def newlayer_par_fast_compiled(Rgg,n,node,other,
    p1,k1,node_nfa,p2,k2,other_nfa,nfa_g):
    '''
    A parallelization-friendly function for solving the trinomial NFA of the agglomerative merging critera
    See:
        F. Cao, J. Delon, A. Desolneux, P. Muse, F. Sur (2004) "An A-contrario approach to hierarchical clustering validity assessment", Diss. INRIA

    Inputs:
        Rgg: The numeber of tests for the trinomial nfa 
        n: The number of point features in total
        node/other: The indicies of the unmerged region in the pandas tree
        p1/p2: The independet uniform probability of an element being in either region (area ratio) 
        k1/k2: The number of points in either region
        node_nfa/other_nfa: binomial nfa of each unmerged region
        nfa_g: binomial nfa of the merged region
    Output:
        node, other, nfa_g: The indices and binomial nfa of the merged region.  nfa_g=-101 if failing merge criteria
    '''
    #Define log-probabilities
    k=[k1,k2]
    p=[p1,p2]
    lp1 = np.log(p1)
    lp2 = np.log(p2)
    lp0 = np.log(1.-p1-p2)
    #For runtime efficacy, limit the size of the array and vectorize in chunks
    splitsize=1000
    range1 = np.arange(k[0],n-k[1],dtype=int)
    taillist=[]
    splitcount = int(np.ceil(len(range1)/splitsize))
    for range1loc in np.array_split(range1,splitcount):
        #Determine the number of number of points in the portion of the trinomial tail in this chunk
        xv,yv = np.meshgrid( range1loc, np.arange(k[1],n-k[0],dtype=float))
        #Remove invlid indices
        xv[yv>=n-xv]=-1.
        data=np.vstack([xv.ravel(),yv.ravel()])
        xv=None; yv=None #temp free memory 
        data=data[:,data.min(axis=0)>0.].T
        #Compute trinomial coefficients using Sterling's approximation (avoid overflow errors)
        N=float(n)
        M=data[:,0].astype(float)
        lcoeff = np.copy(N*np.log(N) - M*np.log(M) - (N-M)*np.log(N-M) + 0.5*(np.log(N)-np.log(M)-np.log(N-M)-np.log(2*np.pi)))
        N=float(n)-data[:,0]
        M=data[:,1].astype(float)
        lcoeff = lcoeff + N*np.log(N) - M*np.log(M) - (N-M)*np.log(N-M) + 0.5*(np.log(N)-np.log(M)-np.log(N-M)-np.log(2*np.pi))
        #Compute trinomial probabilities for each point in the tail 
        lprob = data[:,0]*lp1 + data[:,1]*lp2 + (float(n)-data[:,0]-data[:,1])*lp0
        lprob = lprob + lcoeff
        #Sum this region of the trinomial tail
        try:
            tail = sp.special.logsumexp(lprob)
            taillist.append(np.copy(tail))
        except:
            pass 
    #Compute the entire trinomial tail 
    tail = np.vstack(taillist)
    tail = sp.special.logsumexp(tail)
    #Determine NFA
    nfa_gg = -(tail/np.log(10.))-np.log10(Rgg)
    #Make sure the probability is valid 
    if np.math.isinf(nfa_gg):
        nfa_gg=1001.
    elif np.math.isnan(nfa_gg):
        print('ISNAN')
        quit()
    #Only need to check one merging criteria here (rest in Newlayer function)
    if  nfa_g >= nfa_gg:  ##merge condition
        return node, other, nfa_g
    else:#do not merge condition
        return node, other, -101.

def newlayer(df_tree,layer,NUMPOINTS,lines,step,numleafs,R,subprocess_count):
    '''
    Add a new layer to the hierarchical tree obeying the agglomerative merging criteria in:
        F. Cao, J. Delon, A. Desolneux, P. Muse, F. Sur (2004) "An A-contrario approach to hierarchical clustering validity assessment", Diss. INRIA
    Inputs:
        df_tree: Current hierarchical tree, pandas array
        layer: integer index indicating this layer of the tree
        NUMPOINTS: Total size of data set before reducing lines
        lines: The point features to cluster
        step: The distance between the finest regions, to check adjacency
        R: The number of tests, total possible hyperrectangles of the partitioning 
        subprocess_count: How many parallel process to run in trinomial estimation 
    Outputs:
        df_tree: Added column "layer", with each row given a pair ("index, nfa") an index indicating its branch at this layer, and its binomial nfa
    '''
    #make empty layer for the new branches
    now = '%d'%layer
    newcount=0 #0 reserved for nanidx
    df_tree[now] = pd.Series([(np.nan,np.nan) for x in range(len(df_tree.index))])
    #Get data from the last branch 
    last = '%d'%(layer-1) if layer>1 else 'leaf'
    lastlist = df_tree[last].to_list()
    lastlist = np.array(lastlist)[:,0]
    lastlistU = np.unique(lastlist)
    M = NUMPOINTS
    #Build a matrix for comparing the leafs at this layer to be agglomeratively merged
    nodecount = len(lastlistU)
    nfamatrix = -101*np.ones((nodecount,nodecount))
    #Define the number of binomial and trinomial tests
    Rg=R; Rgg=R*(R-1)/2;
    #Parallelize the trinomial check for merging criteria
    newlayer_par_call = functools.partial(newlayer_par_fast_compiled, Rgg, M)
    iterable = []
    starttime=time.time()
    positives = 0;
    negatives = 0;
    #Iterate over all node pairs 
    for node in lastlistU:
        for other in lastlistU:
            if other<node:
                #avoid repetition to save time, triangular array access
                continue;
            elif node==other: #assign self to list
                #always merge with self in bad scenarios
                #nfaset[other] = node_nfa if node_nfa>-100 else -100
                node_tree = df_tree.loc[lastlist==int(node)]
                node_nfa = node_tree[last].to_list()
                node_nfa = np.array(node_nfa)
                node_nfa = node_nfa[0,1]
                nfaout = node_nfa if node_nfa>-100 else -100
                nfamatrix[int(node),int(other)] = nfaout
                if nfaout>0:
                    positives=positives+1
                else:
                    negatives=negatives+1
            else:
                #Get data of the node pair
                node_tree = df_tree.loc[lastlist==int(node)]
                node_clusters = node_tree['cluster'].to_list()
                node_clusters = list(chain.from_iterable(node_clusters))
                other_tree = df_tree.loc[lastlist==int(other)]
                other_clusters = other_tree['cluster'].to_list()
                other_clusters = list(chain.from_iterable(other_clusters))
                #Get line data of both disjoint clusters and determine closest approach
                XA = lines[np.asarray(node_clusters).astype('int'), :]
                XB = lines[np.asarray(other_clusters).astype('int'), :]
                XAB = sp.spatial.distance.cdist(XA,XB,metric='chebyshev')
                dAB = np.amin(XAB)

                #Compute the prior probability and current line density associated with each region
                nodecubes = len(node_tree)
                p_node = float(nodecubes)/float(numleafs)
                k_node = len(node_clusters)
                node_nfa = node_tree[last].to_list()
                node_nfa = np.array(node_nfa)
                node_nfa = node_nfa[0,1]
                othercubes = len(other_tree)
                p_other = float(othercubes)/float(numleafs)
                k_other = len(other_clusters)
                other_nfa = other_tree[last].to_list()
                other_nfa = np.array(other_nfa)
                other_nfa = other_nfa[0,1]

                #Apply merging critera
                #If the regions are adjacent: 
                if dAB<=step:
                    #Compute the binomial NFA of the merged region, fairly quick 
                    p = p_node + p_other
                    k = k_node + k_other
                    Rg=R; Rgg=R*(R-1)/2;    
                    nfa_g  = ball_nfa(k,M,p,Rg,p*M,M)
                    #If the NFA is greater than both ancenstors...
                    if nfa_g>max(node_nfa,other_nfa):
                        #And the coarse merging critera is passed (to save runtime)...
                        if nfa_g > (-np.log10(.5) + node_nfa + other_nfa):
                            minpts=min(p_node,p_other)*M
                            maxpts=M
                            #And if the two regions are reasonably meaningful...
                            if k_node>minpts and k_other>=minpts and k_node<=maxpts and k_other<maxpts:
                                r1 = float(k_node)/float(M)
                                r2 = float(k_other)/float(M)
                                if (r1!=1 and r1!=0) and (r2!=1 and r2!=0):#Edge case handling
                                    #We should then evaluate the trinoial tail to consider if the joint event is meaningful!
                                    iterable.append([int(node),int(other),
                                        p_node, k_node,node_nfa,  p_other, k_other, other_nfa,nfa_g])
    

    print("Pos/Neg ratio of last run: %d / %d"%(positives,negatives))
    starttime=time.time()-starttime
    print('iterable length:',len(iterable))
    #Run the parallel process to compute trinomial tails
    if len(iterable)>0:
        starttime=time.time()
        parsteps= min(subprocess_count,len(iterable)) #10 #25
        process_count = min(parsteps,cpu_count())
        chunks = max(1,int(len(iterable)/process_count))
        with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
            results=pool.starmap(newlayer_par_call,iterable,chunksize=chunks)
    #Add the parallel results to memmory 
    if len(iterable)>0:
        for r in results:
            try:
                nfamatrix[r[0],r[1]]=r[2]
            except:
                print('results',results)
                print('r',r)
                quit()

        starttime=time.time()-starttime

    newcount=0
    runwhile = True
    #if no off-diagonals, just do diagonal!
    while runwhile:
        #Now, iterate over the matrix and merge PAIRS which have a maximally meaningful NFA 
        idx = np.unravel_index(np.argmax(nfamatrix),nfamatrix.shape)
        nfamax = nfamatrix[idx[0],idx[1]]
        if nfamax>-101: #allow iteration assigning 'singletons'
            new = (np.copy(newcount),np.copy(nfamax))
            df_tree.loc[lastlist==idx[0],now] = [new]
            df_tree.loc[lastlist==idx[1],now] = [new]
            #may only pair every node once, in pairs or singles
            nfamatrix[idx[0],:] = -101
            nfamatrix[idx[1],:] = -101
            nfamatrix[:,idx[0]] = -101
            nfamatrix[:,idx[1]] = -101
            newcount=newcount+1;
        else:
            runwhile=False
    #Now end, noting that we can at most have halved the total number of regions (merging pairs each layer)
    return df_tree

def associate_lines(goodidxfull, linessave,Xlen, shape, postcluster=1, starfilter=4, densityfilter=0.1, folder='.', name='',spreadcorr=1,
    returndensity=False,newonly=False):
    '''
    Given x/y coordinates at each z-frame, 
    perform data association to account for temporal aliasing
    using an a-contrario point alignment detector:
        "An Unsupervised Point Alignment Detection Algorithm"
        by J. Lezama, G. Randall, J.M. Morel and R. Grompone von Gioi,
        Image Processing On Line, 2015. http://dx.doi.org/10.5201/ipol.2015.126
    Inputs: 
        goodidxfull - line indices which were uncluseted
        linessave -   the original line features from 1st-order detection 
        Xlen - number of all 1st order detections (for indexing)
        shape -       [X,Y,Z] dimension of data cube
        postcluster - flag to enable association, otherwise return input
                        Default 1.  Set to 2 for returning goodlines/badlines
        starfilter -  threshold for angular tolerance to reject zero-velocity features 
                        Default 4 for pi/(8*4) rads after z-scaling.  Set to 0 to disable 
        densityfilter- threshold for density rejection of 'inadmissible associations'
                            Default to 0.1 diameter-length ratio.  Set to 0 to disable
        folder -       location to save html of 3d plotly image 
        name  -        name to save html of 3d plotly image
    Outputs:
        templinesFilt -   Associated parametric features,  scaled directional vecotrs 
        lines_full2 -     Associated line features (spatially-located)
                            if postcluster==2, this is 'goodlines'
        goodlinesfull2 -  Unassociated line features (spatially-located)
                            if postcluster==2, this is 'badlines'
        numbeforefilter - number of unclustered and unassociated features, for accounting
    '''

    #print('staring agg.py',end=' ',flush=True)
    #Definitions
    #(padatafd,padatafile)=tempfile.mkstemp(suffix='.txt') #'padata.txt'
    #(paoutfd,paoutfile)=tempfile.mkstemp(suffix='.txt') #'paout.txt'
    padatafile='%s/padata_%s.txt'%(folder,name)
    paoutfile='%s/paout_%s.txt'%(folder,name)
    #print('building vectors',end='...',flush=True)
    aa,bb,cc = shape
    
  
    #build a list of indexes to the 1st-order list
    tempgoodidxfull2 = np.copy(np.asarray(goodidxfull))
    goodidxfull2 = np.zeros((Xlen))
    goodidxfull2[tempgoodidxfull2]=1
    goodidxfull2 = goodidxfull2==1
    #build the list of unclustered lines
    lines_full2 = np.copy(linessave)
    goodlinesfull2 = lines_full2[goodidxfull2]
    lines_full2 = lines_full2[goodidxfull2]
    templinesFilt = np.copy(lines_full2)
    templinesFiltBackup = np.copy(lines_full2)
    #save the number of unclustered lines before association 
    numbeforefilter = len(goodlinesfull2)
    if numbeforefilter==0:
        postcluster=0
    #Build a list of [xyz] coordinates (interpolate trajecotry to each xy, or streak midpoints)
    scaleflag=1.
    if spreadcorr>0:
        spreadfactor = max(1., min(aa,bb)/cc)*int(spreadcorr)
    elif spreadcorr==0:
        spreadfactor= 1.#/float(cc)
    else:
        scaleflag=-1*np.copy(spreadcorr)
        spreadfactor=1
    if postcluster>0:
        #Build a list of association-valid coordinates 
        #We reject points which are very close to vertical (pi/8*4=pi/32),
        #   as they should not suffer from temporal aliasing 
        #We will scale the z-axis s.t. data is closer to a unit cube (without scaling xy)
        #   in order to improve seperability, as in whitening. 
        #   XY scaling complicates the coordinate system transforms and is avoided. 
        
        len2=10000
        starfilter=float(starfilter)
        starangle= 22.5 if starfilter==0 else 22.5/float(starfilter)
        shape2 = [aa,bb,cc*spreadfactor]
        try:
            while starangle<85. and len2>1000:
                shape2 = [aa,bb,cc*spreadfactor]
                objlist = []
                for z in range(cc):
                    for k in range(len(goodlinesfull2)):
                        locline = np.copy(goodlinesfull2[k,:6]).squeeze()
                        locline[5]=locline[5]*spreadfactor
                        locline[2]=locline[2]*spreadfactor
                        k_len=np.linalg.norm(locline[:3]-locline[3:6])
                        el = np.arccos((locline[5]-locline[2])/k_len)
                        el = el-np.pi if el>np.pi else el
                        starcond = True if starfilter==0 else ((el*180./np.pi) > starangle)
                        if starcond: #choose nonvertical
                            zs=float(z)*spreadfactor
                            x,y = interp_frame_xy(locline,zs,double=True,shape=shape2)
                            if not all(np.array([x,y])==0):
                                #if (x>0) and (x<aa) and (y>0) and (y<bb) and (zs>0) and (zs<int(cc*spreadfactor*scaleflag)):
                                objlist.append([y,x,zs])
                                #objlist_unscale.append([y,x,z])

                len2 = len(objlist) #Length of associatable xyz coordates
                if len2>1000:
                    starangle=starangle+5. if starfilter>0 else 90.
        except Exception as e:
            print(e)
            quit()

        if len2==0:
            #print('NO VALID TEMPORALLY ALIASED POINTS, STOPPING ASSOCIATION',flush=True)
            templinesFilt = templinesFiltBackup
            lines = np.copy(templinesFilt)
            postcluster=0
            if returndensity:
                raise Exception('stopping association ')

    #If there are still valid points to associate... 
    if postcluster>0:
        objlist=np.asarray(objlist)

        #Run data association using the a-contrario point alignment algorithm
        #First, update the list to make sure everything is in-bounds 
        if len(objlist)<=1:
            postcluster=0
            if returndensity:
                raise Exception('Too small to align')
        else:
            painput = np.copy(np.asarray(objlist))
            buffer=1.
            idxobj1 = (np.floor(objlist[:,1])>buffer) & (np.ceil(objlist[:,1])<float(aa-buffer))
            idxobj2 = (np.floor(objlist[:,0])>buffer) & (np.ceil(objlist[:,0])<float(bb-buffer))    
            idxobj3 = (np.floor(objlist[:,2])>buffer) & (np.ceil(objlist[:,2])<np.floor((cc*spreadfactor*scaleflag)-buffer))
            idxobj = (idxobj1 & idxobj2) & idxobj3
            if np.count_nonzero(idxobj)==0:
                postcluster=0
                if returndensity:
                    raise Exception('Too small to align')
            else:
                objlist = objlist[idxobj,:]
                try:
                    np.savetxt(padatafile,objlist,fmt='%.3f')
                except Exception as e:
                    print(e)
                    raise Exception ('unable to write padatafile')

                #Specify timeout critera 
                #TODO: THIS IS ONLY DESIGNED FOR TACC, should add a flag to disable this on laptops
                timeoutmin=5 
                if len(objlist)>=1000:
                    timeoutmin=5#10
                elif len(objlist)>=500 and len(objlist)<1000:
                    timeoutmin=5
                elif len(objlist)>=10 and len(objlist)<500:
                    timeoutmin=1#2
                else:
                    timeoutmin=1
                timeoutmin = int(timeoutmin*60)
                raiseflag=False
                #Run with timeout capability using Popen instead of run, more reliable
                try:
                    solve_command="point_alignments_3D %s 0 %d 0 %d 0 %d %s 4"%(padatafile,bb,aa,int(cc*spreadfactor*scaleflag),paoutfile)
                    solve_command = solve_command.split(" ")
                    #solve_result = subprocess.run(solve_command,timeout= timeoutmin)#,
                    solve_result = subprocess.Popen(solve_command, start_new_session=True)
                    solve_result.wait(timeout = timeoutmin)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(solve_result.pid),signal.SIGTERM)
                    print('PTALIGN TIMEOUT')
                    postcluster=0
                    raiseflag=True
                except Exception as e:
                    print(e)
                    postcluster=0
                    raiseflag=True
                if raiseflag and returndensity:
                    raise Exception("PTALIGN ERROR")
                if solve_result.returncode:
                    print(solve_result.returncode)
                    print('ERROR: point_alignments_3D NOT FOUND, TRY RUNNING SETUP.SH')
                    postcluster=0
                    raiseflag=True
                    if  returndensity:
                        raise Exception("point_alignment fails")

    #Load data, handling case of null output
    if postcluster==0:
        templinesFilt=[]
    elif os.stat(paoutfile).st_size==0:
            templinesFilt=[]
    else:
        try:
            templinesFilt = np.loadtxt(paoutfile)
        except:
            templinesFilt=[]

    #Account for zero-associated results, and return the input data
    if (len(templinesFilt)==0) and (not newonly):
        templinesFilt = templinesFiltBackup
        postcluster=0
    elif (len(templinesFilt)>0):
        if templinesFilt.ndim==1:
            templinesFilt = templinesFilt[np.newaxis,:]
        templinesFilt[:,2]=templinesFilt[:,2]/spreadfactor
        templinesFilt[:,5]=templinesFilt[:,5]/spreadfactor
        lines = np.copy(templinesFilt)
        paoutput = np.copy(templinesFilt)
    else: #(len(templinesFilt)==0) and (newonly):
        raise Exception("no alignments and newonly=True.  raising error.  in future, add handling to make extras=all")

    #Input data must be normalized to outplut
    lines = np.copy(templinesFilt)
    k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)

    #Reject associated lines which are not well-fit (10% diameter-to-length ratio)
    #TODO: Update default to 0, already ignoring this most of the time 
    if postcluster>0:
        align_diam = np.copy(lines[:,6]).squeeze()
        if align_diam.ndim==0:
            align_diam=np.asarray([align_diam,])
        density = align_diam/k_len
        density_mask = np.asarray(density<densityfilter)
        if densityfilter==0:
            density_mask[:] = 1
        density_count = np.count_nonzero(density_mask)
        #Control filtering and printing
        if density_count==0:
            #build_xyzplot(np.asarray(objlist_unscale),[],shape,folder=folder, name='%s_associate'%name)
            templinesFilt = templinesFiltBackup
            lines = np.copy(templinesFilt)
            postcluster=0
        else:
            templinesFilt = templinesFilt[density_mask,:]
            lines = lines[density_mask,:]
            paoutput = paoutput[density_mask,:]
            k_len = k_len[density_mask]
            align_diam = align_diam[density_mask]
            #build_xyzplot(np.asarray(objlist_unscale),lines,shape,folder=folder, name='%s_associate'%name)
    else:
        align_diam = np.asarray([])

    #Compute line parameters 
    #lines may be redefined and should be recomputed

    k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
    idx = lines[:,5]<lines[:,2]
    temp = np.copy(lines[idx,:])
    lines[idx,:3]=temp[:,3:6]
    lines[idx,3:6]=temp[:,:3]
    el = np.arccos((lines[:,5]-lines[:,2])/k_len)
    az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
    lines[:,-1] = -lines[:,-1] #lognfa to nlognfa
    lines_full2 = np.copy(lines)
    linesNorm = np.copy(np.array([k_len*np.cos(az)*np.sin(el),k_len,lines[:,6],
                k_len*np.sin(az)*np.sin(el),k_len*np.cos(el),lines[:,-1]]).T)


    #If still associating, remove lines which are accounded for in an alignment cylinder,
    #   and return all other 'unassociated' lines (inc. stars) for outlier detection.
    #   Use alignment_diameter/2 for radial proximity
    #Otherwise define the input data for return
    #Isolate original lines that don't touch lines_full_2
    if postcluster>0:
        try:
            goodlinesfull2 = my_filter(goodlinesfull2,shape,skipper=False, buffer=1.)
            lines_full2 = my_filter(lines_full2,shape,skipper=False, buffer=1.)
            _,extras = remove_matches_block(goodlinesfull2,lines_full2,cc,diam=align_diam) #must spline
        except Exception as e:
            print(e)
            extras=[]
        extras = np.asarray(extras)
        if len(extras)>0:
            try:
                if extras.shape[1]==0:
                    extras=extras.T
            except Exception as e:
                print(e)
        if len(lines_full2):
            try:
                if lines_full2.shape[1]==0:
                    lines_full2=lines_full2.T
            except Exception as e:
                print(e)
        if len(linesNorm):
            try:
                if linesNorm.shape[1]==0:
                    linesNorm=linesNorm.T
            except Exception as e:
                print(e)

        #Format data to return 
        if postcluster==1:
            if (len(extras)>0) and (not newonly):
                try:
                    lines_full2 = np.vstack([lines_full2, extras])
                except Exception as e:
                    print(e)
        elif postcluster==2:
            lines_full2 = lines_full2 #alias to 'goodlines'
            goodlinesfull2 = extras #alias to 'badlines'

        if len(extras)>0:
            lines = np.copy(extras)
            k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
            el = np.arccos((lines[:,5]-lines[:,2])/k_len)
            az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
            try:
                extrasNorm = np.array([k_len*np.cos(az)*np.sin(el),k_len,lines[:,6],
                            k_len*np.sin(az)*np.sin(el),k_len*np.cos(el),lines[:,-1]]).T
            except Exception as e:
                print(e)
        else:
            extrasNorm=np.asarray([])
   
        if (len(extrasNorm)>0) and (not newonly):
            try:
                templinesFilt = np.vstack([linesNorm, extrasNorm])
            except Exception as e:
                print(e)
        else:
            templinesFilt = np.copy(linesNorm)
    else:
        templinesFilt = linesNorm

    #Clean-up
    try:
        os.remove(padatafile)
    except:
        pass
    try:
        os.remove(paoutfile)
    except:
        pass
    if returndensity:
        return templinesFilt, lines_full2, goodlinesfull2, numbeforefilter,align_diam
    return templinesFilt, lines_full2, goodlinesfull2, numbeforefilter


def outlier_alg(templinesFilt, pcanum, xnum, eps, runoutlier=1,name="temp",folder="."):
    '''
    Apply a-contrario outlier rejection assuming a gaussian distribution, use mahanalobis distance for a chi-2 NFA model on individual points.
    Input:
        templinesFilt: The data to process with run_pca and compute outliers from.  
        pcanum/xnum: The dimension of input data and the dimension of the PCA outupt
        runoutlier: A flag to disable processing but return indices
        name/folder: UNUSED, had been used in plotting for debug
    Output:
        goodlines/badlines: the indices of templinesFilt corresponding to meaningful/unmeaningful lines
    '''
    #Get gaussian fit of data
    if templinesFilt.shape[0]<pcanum:
        print('too few lines to cluster, less than xnum',xnum)
        print('passing all lines as good, to keep detections')
        tempgoodidxfull = np.asarray(range(templinesFilt.shape[0]))
        score_samples = np.asarray(range(templinesFilt.shape[0]))
        e2 = -np.inf
        goodidxfull = tempgoodidxfull[score_samples>e2]
        badidxfull = tempgoodidxfull[score_samples<=e2]
        return goodidxfull,badidxfull
    _,XNocluster = run_pca(templinesFilt,templinesFilt,pcanum,xnum)
    if xnum==2 and pcanum==3:
        xnum=3
    if xnum<0:
        xnum=-xnum
    XFilt =np.copy(XNocluster)
    model = RunBayes(1,np.copy(xnum),np.copy(XFilt),wtype='process')

    #Get the mean and covariance data 
    X = np.copy(XNocluster)
    score_samples = -model.score_samples(X) / np.log(10)
    means2 = model.means_.squeeze()#[weightsort]
    covar2 = model.covariances_.squeeze()#[weightsort]
    #If full rank, we would rather use the sample mean and covariance
    print('X shape ',X.shape)
    if np.linalg.matrix_rank(np.cov(X.T))==X.shape[-1]:
        means2 = np.mean(X,axis=0)
        covar2 = np.cov(X.T)
    else:
        print('resorting to scipy covar due to nonsingular sample')
  
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

    #For each line sample, compute the mahanalobis distance and form the NFA using the chi2 tail
    method = 'max'
    for i in range(len(score_samples)):
        mahanrow = ((X[i]-means2).T @  covarinv @ (X[i]-means2))
        mahanrow = cc * mahanrow
        mahan[i] = 1.-crv.cdf(mahanrow)
    score_samples = -np.log10(mahan) - np.log10(len(mahan))

    #Check for meaningfulness
    e2=np.copy(eps)
    eps=e2 #-log10(nfa)<-log10(e=1)
    if runoutlier==0:
        e2=-np.inf
        eps=-np.inf
        print('disabling outlier detection')
    print('min: %.2f, max: %.2f, eps: %.2f'%(min(score_samples),max(score_samples),eps),flush=True)

    #Output results
    tempgoodidxfull = np.asarray(range(len(X)))
    goodidxfull = tempgoodidxfull[score_samples>e2]
    badidxfull = tempgoodidxfull[score_samples<=e2]
    ## PROPOSAL
    #if X.shape[-1]==3:
    #    build_xyzplot(X[badidxfull,:],[], [1.,1.,1.], folder=folder, name='%s_outliers'%(name),ptdetect = X[goodidxfull,:])
    return goodidxfull,badidxfull


     
def detect_outliers(shape,lines=[],folder='',savename='temp',args=None,
    stars=0,e2=0,injectScale=0, subprocess_count=10, postcluster=0,runoutlier=1,spreadcorr=1,avoidrun=False):
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
    #print('Filter Image and Line Data ')
    try:
        aa,bb,cc = shape
    except:
        print('ERROR: EMPTY IMG')
        quit()
    # Format image data 
    #if len(img)!=0:
    #    v = np.array(np.stack((img,)*3,axis=-1))
    #    aa,bb,cc,_ = v.shape
    #else:
    #    print('ERROR: EMPTY IMG')
    #    quit()

    #Format lines into proper matrix shape, and 
    lines = format_lines(lines,aa=aa,filt=0)
    if avoidrun:
        print('AVOIDING RUNNING CLUSTERING ALG...')
        goodlines=lines
        badlines=np.asarray([]).reshape(-1,lines.shape[1])
        np.save('%s/goodlines_%s.npy'%(folder,savename),goodlines)
        np.save('%s/badlines_%s.npy'%(folder,savename),badlines)
        return goodlines,badlines
    eps=np.copy(e2)
    e2=0
    linessave=np.copy(lines)
    print('initializing clustering, linessave shape:',linessave.shape)
    linessavefilter = my_filter(linessave,shape,skipper=False, buffer=1.)
    print('initializing clustering, filtered shape:',linessavefilter.shape)
    #Outlier detection only viable if there exist a 
    # statistically meaninful number of events 

    '''
    ------------------------------------------------------
    '''
    #print('Construct Feature Vectors From Line Data')
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
    #print('Set some variables')
    #Number of PCA components for decomposition
    #unit direction + length on separate axis
    pcanum=3#length, az, el (width and nfa agnostic)
    #Number of PCA components for GMM separation
    xnum = 3;#3#-2 #max(pcanum-2,2)

    '''
    ------------------------------------------------------
    '''
    #print('Separate Training/Testing Data')

    linesFilt = np.copy(linesNorm)
    XFilt,XFull = run_pca(linesFilt,np.copy(linesNorm),pcanum,xnum)        

    '''
    ------------------------------------------------------
    '''
    #print('Proximity Clustering ')
    '''
    ------------------------------------------------------
    '''

    #Number of clusters to test
    print('Xshape',XFull.shape)
    dims = XFull.shape[1]
    #print('dims %d, NTests %.2f, lognfa %.2f'%(dims,Ntests,-np.log10(Ntests))) 
    #print('Computing...')
    starttime=datetime.now()
        
    #parallelization info
    numsteps= subprocess_count
    process_count = min(numsteps,cpu_count())
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
    NUMPOINTSPRE = np.copy(len(templinesFilt))

    nparts=0 
    while runacclust==1:
        counter+=1;

        #Reproject and cluster data
        iterable=[]

        ## ADD NOISE INJECTION TO templinesFilt BEFORE PROJECTION: Uniform
        ## THEN after clustering remove outliers [idx<idxcutoff] 
        ## noise injection on outlier will require additional steps...   
        
        ## ADD NOISE INJECTION
        
        _,XNocluster = run_pca(templinesFilt,templinesFilt,pcanum,xnum)        
        XNoclusterCopy = np.copy(XNocluster)
        if injectScale>0:

            injectLimit = len(XNocluster)
            injectNew = injectScale*injectLimit if injectScale>0 else 2;
            injectLines = np.random.uniform(size=(injectNew, dims));
            XNocluster = np.concatenate((XNocluster,injectLines),axis=0)     

        layer = 0
        #DATA REPROJECTION REDUCES NUM POINTS BUT NOT NUM TESTS
        NUMPOINTS = np.copy(len(XNocluster))

        ## DETERMINE THE NUMBER OF PARITIONS TO USE 
        #Defaults s.t. each partition contains 1 point in Expectation under a uniform independent model 
        #nparts = 100
        d = .01
        if nparts==0:
            if False:
                #'''
                #THIS CAN BE A MASSIVE IMAGE
                #dStore = np.empty((len(XNocluster),len(XNocluster)))
                #dStore[:] = np.nan
                print('Starting smart Dstore',flush=True)
                dStore = np.zeros((len(XNocluster),))
                dTemp = np.ones((len(XNocluster),))
                for i in range(len(XNocluster)-1):
                    dTemp[:]=1
                    for j in range(i+1,len(XNocluster)):
                        #dStore[i,j]=np.linalg.norm(XNocluster[i,:]-XNocluster[j,:],np.inf)
                        dTemp[j]=np.linalg.norm(XNocluster[i,:]-XNocluster[j,:],np.inf)
                    dStore[i] = np.nanmin(dTemp)
                #NOTE: ADDING NOISE INCREASES MEDIAN SUBSTANTIALLY
                # SINCE THE CLUSTERED POINTS HAVE A SURPRISINGLY SMALL SEPARATION
                #  COMPARED TO THE NOISE POINTS, WHICH HAVE NECESSARILY WIDER SPACEING                

                #dStore = np.nanmin(dStore[:,1:],axis=0) #nearest neightbor distances
                dStore = dStore[dStore>0]
                d = np.nanmedian(dStore) #*2. #increase to let be convex
                print('Dmedina:',d)
                d = np.nanmean(dStore) #*2. #increase to let be convex
                print('Dmean:',d)
                dstd = np.nanstd(dStore) #*2. #increase to let be convex
                print('Dstd:',dstd)
                print('D_numroot',1./NUMPOINTS**(1/3.))
                print('nparts =', int(np.ceil(NUMPOINTS**(1/3.))))
                d=d+3.*dstd;#seems ~~dnumroot for starlink AND navstar!
                print('D3std:',d)
                print('nparts =', int(np.ceil(1./d)))
                #'''
            else:
                d=(1./NUMPOINTS)**(1/dims)
            nparts = int(np.ceil(1./d))

        #RUN AGGLOMERATIVE HIERARCHICAL CLUSTERING
        print('NPARTS:',nparts)
        XNocluster = np.copy(XNoclusterCopy)       
        R = (nparts*(nparts+1)/2)**dims
        df_tree,XNocores = makeinit(XNocluster, R, nparts, dims,NUMPOINTSPRE)
        runlayer=1
        allstart = time.time()
        while runlayer==1:
            layer+=1
            start = time.time()
            nleaves = nparts**dims
            R = (nparts*(nparts+1)/2)**dims
            NUMPOINTS=NUMPOINTSPRE
            df_tree = newlayer(df_tree,layer,NUMPOINTS,XNocores,1./nparts,nleaves,R,subprocess_count)
            last = '%d'%(layer-1) if layer>1 else 'leaf'
            now = '%d'%layer

            lastlist = df_tree[last].to_list()
            lastlist = np.array(lastlist)[:,0]
            lastlistU = np.unique(lastlist)
            nowlist = df_tree[now].to_list()
            nowlist = np.array(nowlist)[:,0]
            nowlistU = np.unique(nowlist)

            runlayer = 1 if len(nowlistU)<len(lastlistU) else 0
            print('layertime', time.time() - start)

        print('alltime', time.time() - allstart)
            
        if injectScale>0:
            XNocluster = XNocluster[:injectLimit,:]
        print('XNocluster len: ',len(XNocluster))
        pt1list = []
        balllist = []
        radiuslist = []
        nfalist = []

        #Gather the data from the clustering tree
        for node in nowlistU:
            node_tree = df_tree.loc[nowlist==int(node)]
            node_clusters = node_tree['cluster'].to_list()
            node_clusters = list(chain.from_iterable(node_clusters))
            ## REMOVE NOISE INJECTION
            node_clusters = np.asarray(node_clusters)
            if injectScale>0:
                node_clusters = node_clusters[node_clusters<injectLimit]

            node_nfa = node_tree[now].to_list()
            node_nfa = np.array(node_nfa)
            node_nfa = node_nfa[0,1]

            #extned merged appended lists
            #store now includes disjoint hierarchies!
            #radius is the 'number of partitions' volumetric
            pt1list.append(node)
            balllist.append(node_clusters)
            radiuslist.append(len(node_tree))
            nfalist.append(node_nfa)
        
        #Build  anew tree containing only the maximally meaningful roots, for ease of access
        store = pd.DataFrame(
            {'core': pt1list, 'cluster': balllist, 
            'radius': radiuslist ,   'nfa': nfalist},
            columns=['core','cluster','radius','nfa'])

        print('RUNTIME: ',datetime.now()-starttime)
        
        #Identify maximally meaningfu lcluster
        #clusters are now disjoint
        e=0
        if any(store['nfa'].to_numpy()>e2):
            #get list of items at maximum NFA
            nfastore = store.loc[store['nfa']>e]
            meaningsort = np.argsort(nfastore['nfa'].to_numpy())
            clustmask = np.zeros((len(XNocluster)))
            for ordidx,nfastore_core in enumerate(nfastore['core'].to_numpy()[meaningsort]):
                subclustmask = np.zeros((len(XNocluster)))
                #tempnfastore = nfastore['cluster'].tolist()
                clustlist = nfastore.loc[store['core']==int(nfastore_core),'cluster'].to_numpy()
                for clustidx in clustlist:
                    clustmask[clustidx]=1                                
                    subclustmask[clustidx]=1  
                subclustmask = subclustmask==1                              
                if ordidx==0 and counter==1:
                    CLUST1idxfull=np.append(CLUST1idxfull,goodidxfull[subclustmask])
                elif ordidx==1 and counter==1:
                    CLUST2idxfull=np.append(CLUST2idxfull,goodidxfull[subclustmask])
                elif ordidx==2 and counter==1:
                    CLUST3idxfull=np.append(CLUST3idxfull,goodidxfull[subclustmask])
                else:
                    CLUSTXidxfull=np.append(CLUSTXidxfull,goodidxfull[subclustmask])

            #Assign clusters to storage
            clustmask = clustmask==1
            XCluster = XNocluster[clustmask,:]
            XAllClusters.append(XCluster)
            XNocluster = XNocluster[~clustmask,:]
            goodidxfull = goodidxfull[~clustmask]
            templinesFilt = templinesFilt[~clustmask,:]
            print('cluster size B: ', np.count_nonzero(clustmask))
            ## PROPOSAL
            #build_xyzplot(XNocluster,[], [1,1,1], folder=folder, name='%s_cluster%d'%(savename,counter),ptdetect = XCluster)

        else:
            runacclust=0
        if len(XNocluster)<=1:
            runacclust=0

    #Data Association prior to outlier detection (for temporal aliasing)
    print('PREASSOC SIZE',templinesFilt.shape)
    #Data Association prior to outlier detection (for temporal aliasing)
    print('input shape to assoc.:',shape)
    print('before assoc, linessave shape:',linessave.shape)
    linessavefilter = my_filter(linessave,shape,skipper=False,buffer=1.)
    print('before assoc, filtered shape:',linessavefilter.shape)
    templinesFilt,lines_full2,goodlinesfull2,numbeforefilter = associate_lines(goodidxfull,linessave,len(XFull),shape,postcluster,folder=folder,name=savename,spreadcorr=spreadcorr)
    print('after assoc, lines_full2 shape:',lines_full2.shape)
    linessavefilter = my_filter(lines_full2,shape,skipper=False,buffer=1.)
    print('after assoc, filtered shape:',linessavefilter.shape)
    #the goodlines are fine... we just need to actually print the right "badlines"
    assocParam = templinesFilt
    assocLines = lines_full2
    unassocLines = goodlinesfull2
    #INDICIES DO NOT CARRY OVER DUE TO ASSOCIATION STRUCTURE
    print('POSTASSOC SIZE',templinesFilt.shape)
    #clustering only the remainders... goodidxfull should be in relation to...
    #will lose earlier bad lines either way!!!
    goodidxfull,_ = outlier_alg(templinesFilt, pcanum, xnum, eps, runoutlier,name=savename,folder=folder)
    print('first goodidxfull',len(goodidxfull))

    #CORRECTION: GOODLINES IS "joined", BADLINES is "REJECTED"
    #use Linesfull2 as access.  REMAINALL holds pre-joined data

    #initialize
    #TEMP BECOMES THE FILTER
    #booleanize
    tempgoodidxfull = np.copy(np.asarray(goodidxfull))
    goodidxfull = np.zeros((len(templinesFilt)))
    goodidxfull[tempgoodidxfull]=1
    goodidxfull = goodidxfull==1
    #access
    #the associated line sets accounts for replacement of unassociated lines
    #and is practically all 'unclusterable'
    #of the associated line set, these are 'meaningful outliers'
    goodlinesfull = lines_full2[goodidxfull]
    #of the associated line set, these are 'unmeaningful - unclusterable yet noise"
    badlinesfull = lines_full2[~goodidxfull]
    #merge the noise with those features with were never associated - ARE clusters!
    #badlinesfull = np.vstack((meaningfulclusters, badlinesfull))


    ##CHOOSING ONLY FROM THE ASSOCIATED LINE FEATURES
    OUTLIERlinesfull =  goodlinesfull;
    REMAINlinesfull = badlinesfull;
    REMAINALLlinesfull = goodlinesfull2;


    lines_full = np.copy(linessave)
    CLUSTALLidx    = np.zeros((len(XFull))) #XFULL replaced
    CLUST1idxfull2 = np.zeros((len(XFull))) #XFULL replaced
    CLUST2idxfull2 = np.zeros((len(XFull)))
    CLUST3idxfull2 = np.zeros((len(XFull)))
    CLUSTXidxfull2 = np.zeros((len(XFull)))
    CLUST1idxfull = np.asarray(CLUST1idxfull.flatten(),dtype=int)
    CLUST2idxfull = np.asarray(CLUST2idxfull.flatten(),dtype=int)
    CLUST3idxfull = np.asarray(CLUST3idxfull.flatten(),dtype=int)
    CLUSTXidxfull = np.asarray(CLUSTXidxfull.flatten(),dtype=int)
    CLUST1idxfull2[CLUST1idxfull]=1
    CLUST2idxfull2[CLUST2idxfull]=1
    CLUST3idxfull2[CLUST3idxfull]=1
    CLUSTXidxfull2[CLUSTXidxfull]=1
    CLUST1idxfull = CLUST1idxfull2==1
    CLUST2idxfull = CLUST2idxfull2==1
    CLUST3idxfull = CLUST3idxfull2==1
    CLUSTXidxfull = CLUSTXidxfull2==1
    CLUSTALLidx   = ((CLUST1idxfull | CLUST2idxfull) | CLUST3idxfull) | CLUSTXidxfull

    badlinesfull = np.vstack((lines_full[CLUSTALLidx], badlinesfull))
    print('Total lines: %d, Unclustered %d, Grouped %d, Accepted: %d, Rejected: %d'%(len(lines_full), numbeforefilter, len(lines_full2),
        len(goodlinesfull), len(badlinesfull)))


    ######################################################################################

    np.save('%s/goodlines_%s.npy'%(folder,savename),goodlinesfull)
    np.save('%s/badlines_%s.npy'%(folder,savename),badlinesfull)

    np.save('%s/CLUST1_%s.npy'%(folder,savename),lines_full[CLUST1idxfull])
    np.save('%s/CLUST2_%s.npy'%(folder,savename),lines_full[CLUST2idxfull])
    np.save('%s/CLUST3_%s.npy'%(folder,savename),lines_full[CLUST3idxfull])
    np.save('%s/CLUSTX_%s.npy'%(folder,savename),lines_full[CLUSTXidxfull])

    np.save('%s/REMAIN_%s.npy'%(folder,savename),REMAINlinesfull)
    np.save('%s/REMAINALL_%s.npy'%(folder,savename),REMAINALLlinesfull)
    np.save('%s/OUTLIER_%s.npy'%(folder,savename),OUTLIERlinesfull)
    ######################################################################################


    print('Done!\n\n')
    return goodlinesfull, badlinesfull

