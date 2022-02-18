'''
PyCIS - Python Computational Inference from Structure

pylib/detect_outliers.py: 2nd-order classification by clustering and outlier rejection
    agglomerative heirarchical clustering method

TODO:
  Clean up indexing and variable passing

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
import functools
import cv2
from pylib.print_detections import interp_frame, interp_frame_xy, build_xyzplot
import subprocess
import plotly.express as px
import plotly.graph_objects as go




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
    failflag=0
    for x in range(linesNorm.shape[1]):
        if np.all(linesNorm[:,x]==linesNorm[0,x]):
            #print("ERROR: detect_outliers: all elements on linesNorm index %d are identical"%x,flush=True)
            #print("\t this may be due to, say, only detecting spatial features with elevation pi/2",flush=True)
            #quit()
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
    '''
    Format lines for matrix processing
    '''
    #Convert line vector to feature matrix 
    l = np.size(np.asarray(lines))/a
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

def my_multinomial_logsf_backup(k,n,p):
    ''' OBSOLETE '''
    s = 0
    flag = 0
    p1,p2 = p
    x=[]
    mylen = 0

    for i in range(k[0],n-1-k[1]):
        mylen=mylen+(n-i-k[1])
    mycounter=0
    x = np.empty((mylen,3))

    for i in range(k[0],n-1-k[1]):
        y = np.empty((n-i-k[1],3))
        y[:,0] = i;
        y[:,1] = np.arange(k[1], n-i)
        y[:,2] = n-i-y[:,1]
        ylen = len(y)
        x[mycounter:mycounter+ylen,:]=y
        mycounter=mycounter+ylen

    try:
        lpmf = stats.multinomial.logpmf(x,n=n,p=[p1,p2,1.-p1-p2])
    except Exception as e:
        print('X', x)
        print('K',k)
        print('n',n)
        print('ps:',(p1,p2,1-p1-p2))
        print(e)
        quit()
    s = sp.special.logsumexp(lpmf)
    return s

def my_multinomial_logsf_comb(k,n,p):
    ''' Compute the trinomial joint tail sf for [k1,k2] up to n with [p1,p2] probability '''
    s = 0
    flag = 0
    p1=p[0]
    p2=p[1]
    x=[]
    mylen = 0

    lp1 = np.log(p1)
    lp2 = np.log(p2)
    lp0 = np.log(1.-p1-p2)
    tail = np.nan
    #COMPUTE TRINOMIAL COEFF (not multinomial)
    #VECTORIZING might save time
    elements = int((k[0]+k[1]-n-1)*(k[0]+k[1]-n)/2)
    data = np.zeros((elements,2 ))
    counter=0
    for i in range(k[0],n-k[1]):
        for j in range(k[1], n-i):
            data[counter,:]=[int(i),int(j)]
            counter=counter+1
    lcoeff = np.log(sp.special.comb(int(n),data[:,0]).astype('float'))
    lcoeff = np.log(sp.special.comb(int(n)-data[:,0],data[:,1]).astype('float')) + lcoeff
    data = data.astype('float')
    lprob = data[:,0]*lp1 + data[:,1]*lp2 + (float(n)-data[:,0]-data[:,1])*lp0
    lprob = lprob + lcoeff
    tail = sp.special.logsumexp(lprob)
    return tail

def my_multinomial_logsf(k,n,p):
    ''' 
    Compute the trinomial joint tail sf for [k1,k2] up to n with [p1,p2] probability 
    Use fast array construction and the Sterling approximation of the trinomial coefficient
    '''
    s = 0
    flag = 0
    p1=p[0]
    p2=p[1]
    x=[]
    mylen = 0

    lp1 = np.log(p1)
    lp2 = np.log(p2)
    lp0 = np.log(1.-p1-p2)
    tail = np.nan
    #COMPUTE TRINOMIAL COEFF (not multinomial)
    #VECTORIZING might save time
    #Define data indices
    xv,yv = np.meshgrid( np.arange(k[0], n-k[1],dtype=float), np.arange(k[1],n-k[0],dtype=float))
    xv[yv>=n-xv]=-1.
    data=np.vstack([xv.ravel(),yv.ravel()])
    data=data[:,data.min(axis=0)>0.].T
    #Compute trinomial coefficients using Sterling's approximation (avoid overflow)
    N=float(n)
    M=data[:,0]
    lcoeff = N*np.log(N) - M*np.log(M) - (N-M)*np.log(N-M) + 0.5*(np.log(N)-np.log(M)-np.log(N-M)-np.log(2*np.pi))
    lcoeff=np.copy(lcoeff)
    N=float(n)-data[:,0]
    M=data[:,1]
    lcoeff = lcoeff + N*np.log(N) - M*np.log(M) - (N-M)*np.log(N-M) + 0.5*(np.log(N)-np.log(M)-np.log(N-M)-np.log(2*np.pi))
    #Compute tail probability 
    lprob = data[:,0]*lp1 + data[:,1]*lp2 + (float(n)-data[:,0]-data[:,1])*lp0
    lprob = lprob + lcoeff
    tail = sp.special.logsumexp(lprob)
    return tail


def ball_nfa_2(k1,k2,n,p1,p2,Nt,minpts,maxpts):
    ''' compute nfa of cluster '''
    #Ntests = (Nt *(Nt-1.))/2.
    Ntests=Nt
    if k1<=minpts or k2<minpts:
        nfa=-101.
    elif k1>=maxpts or k2>=maxpts:
        nfa=-101.
    else:
        r1 = float(k1)/float(n)
        r2 = float(k2)/float(n)
        r0 = 1.-r1-r2
        p0 = 1.-p1-p2
        if (r1==1 or r1==0) or (r2==1 or r2==0):
            nfa=-101.
        else:
            #tail = stats.multinomial.logsf(k=[k1,k2],n=n,p=[p1,p2])
            tail = my_multinomial_logsf(k=[k1,k2],n=n,p=[p1,p2])
            nfa = -(tail/np.log(10.))-np.log10(Ntests)
            #if nfa>200: #use moment generating estimate bounds
            #    #Kulback-Liebler Divergence for Large Deviation 
            #    #Moment generating tail, Agrawal, R. (2020) Finite-Sample Concentration of the Multinomial in Relative Entropy
            #    tail = -n*( r1*np.log(r1/p1) + r2*np.log(r2/p2) + r0*np.log(r0/p0))
            #    nfa = -(tail/np.log(10.))-np.log10(Ntests)
            if np.math.isinf(nfa):
                nfa=1001.
            elif np.math.isnan(nfa):
                print('ISNAN')
                quit()
    return nfa     

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
            nfa = ball_nfa(k,n,p,Ntests,minpts,maxpts)
            pt1list.append(pt1)
            balllist.append(ball[pt1])
            radiuslist.append(radius)
            nfalist.append(nfa)

    return [pt1list,balllist,radiuslist,nfalist]



def makeinit(lines, R,nparts, dims, NUMPOINTS):
    step=1./nparts
    Ntests=R
    #Ntests = int(nparts**dims)
    n = NUMPOINTS #len(lines)
    p = step**dims #==1/numcubes
    minpts = max(int(np.ceil(p*n)+1),2) #uniform case is lower bound
    maxpts = 1.* n 
    #Ntests is regions while n is pts, should have Ntests<npts
    print('INIT: nparts %d, Ntests %d, n %d, p %.2e, minpts %d, matplts %d'%(nparts,Ntests,n,p,minpts,maxpts))    
    
    cores = np.indices(tuple(np.asarray(np.ones(dims)*(nparts),dtype=int)))
    cores = cores.T.reshape(-1,dims)
    cores = (cores+1.)*step - step/2 #normalize to unit cube

    linecores = np.zeros_like(lines)
    linetree = sp.spatial.KDTree(np.copy(lines))
    coretree = sp.spatial.KDTree(np.copy(cores))
    balls = coretree.query_ball_tree(linetree,r=step/2., p=np.inf)
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
        #if k>minpts:
        #    print('k,n,p,N,nfa:',[k,n,p,Ntests,nfa])
        pt1list.append(ptcount)
        balllist.append(balls[pt1])
        nfalist.append(nfa)
        ptcount=ptcount+1
  
    test = np.asarray(nfalist)
    #test = test[test!=-101.0]
    #print('num non-101 nfas: ',test.shape) 
    #tree to track nfa layers for cluster idx of orig poins
    df_tree = pd.DataFrame(
        {'leaf': list(zip(pt1list,nfalist)), 'cluster': balllist},
        columns=['leaf','cluster']) #append node: zip(layeridx,nfa)i
    nfatest = df_tree['leaf'].to_list()
    nfatest = np.array(nfatest)[:,1]
    return df_tree,linecores 

def newlayer_par(df_tree, lastlist, numleafs,last, now, lines, R, M, step,node, other):
    #print([node,other])
    nfaout = -101. 
    #Get node information 
    node=int(node)
    node_tree = df_tree.loc[lastlist==node]
    nodecubes = len(node_tree)
    p_node = nodecubes/numleafs
    node_clusters = node_tree['cluster'].to_list()
    node_clusters = list(chain.from_iterable(node_clusters))
    k_node = len(node_clusters)
    node_nfa = node_tree[last].to_list()
    node_nfa = np.array(node_nfa)
    node_nfa = node_nfa[0,1]
        
    #collect other node, not self
    other=int(other)
    other_tree = df_tree.loc[lastlist==other]
    #get node information
    othercubes = len(other_tree)
    p_other = othercubes/numleafs 
    other_clusters = other_tree['cluster'].to_list()
    other_clusters = list(chain.from_iterable(other_clusters))
    k_other = len(other_clusters)
    #CHECK FOR ADJACENCY 
    dAB = 1.;
    #lines = Xnocores = linecores

    #finish gathering properties 
    #p is a volumetric fraction, doesn't care about points
    p = (nodecubes+othercubes)/numleafs
    k = k_node + k_other
    other_nfa = other_tree[last].to_list()
    other_nfa = np.array(other_nfa)
    other_nfa = other_nfa[0,1]
    #Rg is 'size of new combined clusters'
    #Rg = numleafs / (nodecubes+othercubes)
    #R1 = numleafs / (nodecubes)
    #R2 = numleafs / (nodecubes)
    #Rgg = R1*(R2-1)/2 if R1==R2 else R1*R2/2
    Rg=R; Rgg=R*(R-1)/2;
    #print('nodes %d, others %d, net %d, all %d'%(nodecubes,othercubes,nodecubes+othercubes,numleafs))
    #print('R1 %.1f, R2 %.1f, Rg %.1f, Rgg %.1f'%(R1,R2,Rg, Rgg))
    nfa_g  = ball_nfa(k,M,p,Rg,p*M,M)
    #logspace conditionsi
    #merge only, don't seek to define e-meaningful 
    #if nfa_g<0:
    #    continue
    if nfa_g<=max(node_nfa,other_nfa):
        return node, other, nfaout
    if nfa_g <= (-np.log10(.5) + node_nfa + other_nfa): 
        return node, other, nfaout
    #R1 is 'size of exising cluster'
    #R2 is 'size of other cluster, removing existing if same'
    #Rgg = R1*(R-1)/2
    nfa_gg = ball_nfa_2(k_node,k_other,M,
        p_node,p_other,Rgg,min(p_node,p_other)*M,M)
    #logspace conditions
    if  nfa_g >= nfa_gg:  ## ancestors and decendents implicit 
        nfaout = nfa_g
    return node, other, nfaout

def newlayer_par_fast(Rgg, M,
    node, other,
    p_node, k_node, node_nfa, p_other, k_other, other_nfa, nfa_g):
    #ONLY PARALLELIZE THE TRINOMIAL COMPUTATIONS
    #startT=time.time()

    nfa_gg = ball_nfa_2(k_node,k_other,M, p_node,p_other,Rgg,min(p_node,p_other)*M,M)

    #logspace conditions
    if  nfa_g >= nfa_gg:  ## ancestors and decendents implicit
        #merge condition
        return node, other, nfa_g
    else:
        #do not merge condition
        return node, other, -101.


def newlayer(df_tree,layer,NUMPOINTS,lines,step,numleafs,R,subprocess_count):
    
    #make empty layers
    now = '%d'%layer
    newcount=0 #0 reserved for nanidx
    df_tree[now] = pd.Series([(np.nan,np.nan) for x in range(len(df_tree.index))])
    #df_tree = df_tree.assign(now=(None,None))
    last = '%d'%(layer-1) if layer>1 else 'leaf'
    lastlist = df_tree[last].to_list()
    lastlist = np.array(lastlist)[:,0]
    lastlistU = np.unique(lastlist)
    #numleafs = len(last)
    #numleafs = len(df_tree)
    #print(df_tree)
    #R = len(lastlist) #possible nodes to pari    
    #R1 = members of i-nodes
    #R2 = members of i-nodes
    M = NUMPOINTS
    #for each unique node
    #print(lastlist)
    #print(lastlist)
    #print('lentree',len(df_tree))
    #quit()
    nodecount = len(lastlistU)
    nfamatrix = -101*np.ones((nodecount,nodecount))

    #newlayer_par_call = lambda node, other: newlayer_par(node, other, df_tree, lastlist, numleafs,last, now, lines, R, M)
    #newlayer_par_call = functools.partial(newlayer_par, df_tree, lastlist, numleafs,last, now, lines, R, M,step)
    Rg=R; Rgg=R*(R-1)/2;
    newlayer_par_call = functools.partial(newlayer_par_fast, Rgg, M)

    iterable = []
    oldpossible = 0
    starttime=time.time()
    positives = 0;
    negatives = 0;
    for node in lastlistU:
        for other in lastlistU:
            if other<node:
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
                node_tree = df_tree.loc[lastlist==int(node)]
                node_clusters = node_tree['cluster'].to_list()
                node_clusters = list(chain.from_iterable(node_clusters))
                other_tree = df_tree.loc[lastlist==int(other)]
                other_clusters = other_tree['cluster'].to_list()
                other_clusters = list(chain.from_iterable(other_clusters))
                XA = lines[np.asarray(node_clusters).astype('int'), :]
                XB = lines[np.asarray(other_clusters).astype('int'), :]
                XAB = sp.spatial.distance.cdist(XA,XB,metric='chebyshev')
                dAB = np.amin(XAB)
                oldpossible=oldpossible+1

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
                #avoid recomputations

                if dAB<=step:
                    p = p_node + p_other
                    k = k_node + k_other
                    Rg=R; Rgg=R*(R-1)/2;
                    nfa_g  = ball_nfa(k,M,p,Rg,p*M,M)
                    #note nfa matrix is already -101 instantiated.
                    if nfa_g>max(node_nfa,other_nfa):
                        if nfa_g > (-np.log10(.5) + node_nfa + other_nfa):
                            iterable.append([int(node),int(other),
                                p_node, k_node,node_nfa,  p_other, k_other, other_nfa,nfa_g])
    
    #print("EXPLORED POSSIBILITIES: ",len(iterable))
    #print("WITHOUT ADJACENCY: ",oldpossible)
    print("Pos/Neg ratio of last run: %d / %d"%(positives,negatives))
    starttime=time.time()-starttime
    #print("LOADING TIME: ",starttime)

    if len(iterable)>0:
        starttime=time.time()
        parsteps= min(subprocess_count,len(iterable)) #10 #25
        process_count = min(parsteps,cpu_count())
        chunks = min(1,int(len(iterable)/process_count))
        #print("Parstep, process, chunk: ",[parsteps,process_count,chunks])
        #print("iterable:",iterable)
        with get_context("spawn").Pool(processes=process_count,maxtasksperchild=1) as pool:
            results=pool.starmap(newlayer_par_call,iterable,chunksize=chunks)
    
    if len(iterable)>0:
        for r in results:
            try:
                nfamatrix[r[0],r[1]]=r[2]
            except:
                print('results',results)
                print('r',r)
                #print('r[0]',r[0])
                #print('r[1]',r[1])
                #print('r[2]',r[2])
                quit()

        starttime=time.time()-starttime
        #print("RESULT TIME: ",starttime)

    newcount=0
    runwhile = True
    #if no off-diagonals, just do diagonal!
    while runwhile:

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


    #print('END ADDLAYER')
    #quit()
    return df_tree

def associate_lines(goodidxfull, linessave,Xlen, shape, postcluster=1, starfilter=4, densityfilter=0.1, folder='.', name='',spreadcorr=1):
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

    #Definitions
    padatafile='padata.txt'
    paoutfile='paout.txt'
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

    #Build a list of [xyz] coordinates (interpolate trajecotry to each xy, or streak midpoints)
    if postcluster>0:
        #Build a list of all input coordinates 
        objlist = []
        for z in range(cc):
            for k in range(len(goodlinesfull2)):
              locline = goodlinesfull2[k,:6].squeeze()
              k_len=np.linalg.norm(locline[:3]-locline[3:6])
              el = np.arccos((locline[2]-locline[5])/k_len)
              if 1==1: #consier all features
                  x,y = interp_frame_xy(locline,z,double=True)
                  if not all(np.array([x,y])==0):
                      objlist.append([y,x,z])
        len1 = len(objlist) #Length of all xyz coordinates
        build_xyzplot(np.asarray(objlist), goodlinesfull2, shape,folder=folder, name='%s_preassociate'%name)
        
        #Build a list of association-valid coordinates 
        #We reject points which are very close to vertical (pi/8*4=pi/32),
        #   as they should not suffer from temporal aliasing 
        #We will scale the z-axis s.t. data is closer to a unit cube (without scaling xy)
        #   in order to improve seperability, as in whitening. 
        #   XY scaling complicates the coordinate system transforms and is avoided. 
        spreadfactor = max(1., min(aa,bb)/cc)*int(spreadcorr)
        len2=10000
        starfilter=float(starfilter)
        starangle= 22.5 if starfilter==0 else 22.5/float(starfilter)
        shape2 = [aa,bb,cc*spreadfactor]
        while starangle<85. and len2>1000:
            shape2 = [aa,bb,cc*spreadfactor]
            print('association z-spreadfactor:',spreadfactor)
            print('starfilter, densityfilter:',[starfilter,densityfilter])
            objlist = []
            objlist_unscale=[]
            for z in range(cc):
                for k in range(len(goodlinesfull2)):
                  locline = np.copy(goodlinesfull2[k,:6]).squeeze()
                  locline[5]=locline[5]*spreadfactor
                  locline[2]=locline[2]*spreadfactor
                  #distxy=((locline[0]-locline[3])**2.+(locline[1]-locline[4])**2.)**.5
                  k_len=np.linalg.norm(locline[:3]-locline[3:6])
                  el = np.arccos((locline[5]-locline[2])/k_len)
                  el = el-np.pi if el>np.pi else el
                  starcond = True if starfilter==0 else ((el*180./np.pi) > starangle)
                  if starcond: #choose nonvertical
                      zs=float(z)*spreadfactor
                      x,y = interp_frame_xy(locline,zs,double=True,shape=shape2)
                      if not all(np.array([x,y])==0):
                          objlist.append([y,x,zs])
                          objlist_unscale.append([y,x,z])

            len2 = len(objlist) #Length of associatable xyz coordates
            print('We consider %d of %d xyz coordinates for temporal aliasing, angle tol %.2f, spread %.2f'%(len2,len1,starangle, spreadfactor),flush=True)
            if len2>1000:
                starangle=starangle+5. if starfilter>0 else 90.
        if len2==0:
            print('NO VALID TEMPORALLY ALIASED POINTS, STOPPING ASSOCIATION')
            templinesFilt = templinesFiltBackup
            lines = np.copy(templinesFilt)
            postcluster=0
            quit()

    if postcluster>0:

        #Run data association using the a-contrario point alignment algorithm
        painput = np.copy(np.asarray(objlist))
        print('max ',np.amax(painput,0))
        print('min ', np.amin(painput,0))
        print('size ',[aa,bb,cc*spreadfactor])
        np.savetxt(padatafile,objlist,fmt='%.3f')
        solve_command='point_alignments_3D %s 0 %d 0 %d 0 %d %s 4'%(padatafile,aa,bb,cc*spreadfactor,paoutfile)
        solve_result = subprocess.run(solve_command,shell=True)#,
        if solve_result.returncode:
            print(solve_result.returncode)
            print('ERROR: point_alignments_3D NOT FOUND, TRY RUNNING SETUP.SH')
            quit()
        try:
            templinesFilt = np.loadtxt(paoutfile)
        except:
            templinesFilt=[]

        #Account for zero-associated results, and return the input data
        if len(templinesFilt)==0:
            build_xyzplot(np.asarray(objlist_unscale),[],shape,folder=folder, name='%s_associate'%name)
            templinesFilt = templinesFiltBackup
            postcluster=0
        else:
            if templinesFilt.ndim==1:
                templinesFilt = templinesFilt[np.newaxis,:]
            templinesFilt[:,2]=templinesFilt[:,2]/spreadfactor
            templinesFilt[:,5]=templinesFilt[:,5]/spreadfactor
            lines = np.copy(templinesFilt)
            paoutput = np.copy(templinesFilt)

    #Input data must be normalized to outplut
    lines = np.copy(templinesFilt)
    k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
    #Reject associated lines which are not well-fit (10% diameter-to-length ratio)
    if postcluster>0:
        align_diam = np.copy(lines[:,6]).squeeze()
        if align_diam.ndim==0:
            align_diam=np.asarray([align_diam,])
        density = align_diam/k_len
        print('DENSITY: ',density)
        density_mask = np.asarray(density<densityfilter)
        if densityfilter==0:
            density_mask[:] = 1
        density_count = np.count_nonzero(density_mask)
        print('valid density (ratio less than threshold) lines: ',density_count)
        #Control filtering and printing
        if density_count==0:
            build_xyzplot(np.asarray(objlist_unscale),[],shape,folder=folder, name='%s_associate'%name)
            templinesFilt = templinesFiltBackup
            lines = np.copy(templinesFilt)
            postcluster=0
            print('0 valid density: setting templinesFilt shape',templinesFilt.shape)
        else:
            templinesFilt = templinesFilt[density_mask,:]
            lines = lines[density_mask,:]
            paoutput = paoutput[density_mask,:]
            k_len = k_len[density_mask]
            align_diam = align_diam[density_mask]
            build_xyzplot(np.asarray(objlist_unscale),lines,shape,folder=folder, name='%s_associate'%name)

        
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
    if postcluster>0:
        fx = len(goodlinesfull2) #input
        fy = len(lines_full2) #reduced
        distMat = np.zeros((fx,fy))
        for i in range(fx):
            for j in range(fy):
              dist=np.nan*np.zeros((cc,))
              for z in range(cc):
                  locline = goodlinesfull2[i,:6].squeeze()
                  x1,y1 = interp_frame_xy(locline,z,double=True,extrap=True)
                  if not all(np.array([x1,y1])==0):
                      locline = lines_full2[j,:6].squeeze()
                      x2,y2 = interp_frame_xy(locline,z,double=True,extrap=True)
                      if not all(np.array([x2,y2])==0):
                          disti=((x1-x2)**2. + (y1-y2)**2.)**.5
                          dist[z] = disti if (disti>=align_diam[j]/2.) else 0.
              distMat[i,j] = np.nanmin(dist)
        #Mask the assocated lines, either to 1 pixel or cylinder diameter
        mask=np.asarray(distMat>1.)
        unmatched= np.asarray(mask.min(axis=1)>0.)
        matches = np.count_nonzero(distMat<1.)
        print('num 1-pixel match %d from shape %d/%d'%(matches,fx,fy))
        extras = goodlinesfull2[unmatched, :]
        if postcluster==1:
            print('postcluster 1: returning associated and unassociated lines')
            lines_full2 = np.vstack([lines_full2, extras])
        elif postcluster==2:
            print('postcluster 2: returning goodlines/badlines')
            lines_full2 = lines_full2 #alias to 'goodlines'
            goodlinesfull2 = extras #alias to 'badlines'

        lines = np.copy(extras)
        k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
        el = np.arccos((lines[:,5]-lines[:,2])/k_len)
        az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
        extrasNorm = np.array([k_len*np.cos(az)*np.sin(el),k_len,lines[:,6],
                    k_len*np.sin(az)*np.sin(el),k_len*np.cos(el),lines[:,-1]]).T
        print('linesNorm shape',linesNorm.shape)
        print('extrasNorm shape',extrasNorm.shape)
        templinesFilt = np.vstack([linesNorm, extrasNorm])
    else:
        templinesFilt = linesNorm
    print('templinesFilt shape',templinesFilt.shape)
    print('closing assoc',flush=True)

    return templinesFilt, lines_full2, goodlinesfull2, numbeforefilter


def outlier_alg(templinesFilt, pcanum, xnum, eps, runoutlier=1,name="temp",folder="."):
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
    XFilt =np.copy(XNocluster)
    model = RunBayes(1,np.copy(xnum),np.copy(XFilt),wtype='process')
    X = np.copy(XNocluster)
    predict_proba = model.predict_proba(X)
    score_samples = -model.score_samples(X) / np.log(10)
    #select rank r<k data
    predict_proba2 = predict_proba.squeeze()#2[:,weightsort]
    weights2 = model.weights_.squeeze()#[weightsort]
    means2 = model.means_.squeeze()#[weightsort]
    covar2 = model.covariances_.squeeze()#[weightsort]

    print('X shape ',X.shape)
    #print('runbayes mean ',means2)
    #print('sample mean ',np.mean(X,axis=0))
    #print('runbayes cov ',covar2)
    #print('sample cov ',np.cov(X.T))
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
    #for each line sample...
    method = 'max'
    for i in range(len(score_samples)):
        mahanrow = ((X[i]-means2).T @  covarinv @ (X[i]-means2))
        mahanrow = cc * mahanrow
        mahan[i] = 1.-crv.cdf(mahanrow)

    score_samples = -np.log10(mahan) - np.log10(len(mahan))


    e2=np.copy(eps)
    eps=e2 #-log10(nfa)<-log10(e=1)
    if runoutlier==0:
        e2=-np.inf
        eps=-np.inf
        print('disabling outlier detection')

    #print('DISABLING OUTLIERS')
    #e2=-np.inf
    #if len(score_samples)>10:
    #    print(score_samples[:10])
    #else:
    #    print(score_samples)
    print('min: %.2f, max: %.2f, eps: %.2f'%(min(score_samples),max(score_samples),eps),flush=True)
    ######################################################################################
    tempgoodidxfull = np.asarray(range(len(X)))
    goodidxfull = tempgoodidxfull[score_samples>e2]
    badidxfull = tempgoodidxfull[score_samples<=e2]


    ## PROPOSAL
    if X.shape[-1]==3:
        build_xyzplot(X[badidxfull,:],[], [1.,1.,1.], folder=folder, name='%s_outliers'%(name),ptdetect = X[goodidxfull,:])



    return goodidxfull,badidxfull


     
def detect_outliers(shape,lines=[],folder='',savename='temp',args=None,
    stars=0,e2=0,injectScale=0, subprocess_count=10, postcluster=0,runoutlier=1,spreadcorr=1):
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
    eps=np.copy(e2)
    e2=0
    linessave=np.copy(lines)
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

        #print('\n\nInjectScale',injectScale)
        #print('xshape',XNocluster.shape)
        #print('rang0:',[np.amin(XNocluster[:,0]),np.amax(XNocluster[:,0])])
        #print('rang1:',[np.amin(XNocluster[:,1]),np.amax(XNocluster[:,1])])
        #print('rang2:',[np.amin(XNocluster[:,2]),np.amax(XNocluster[:,2])])
        layer = 0
        #DATA REPROJECTION REDUCES NUM POINTS BUT NOT NUM TESTS
        NUMPOINTS = np.copy(len(XNocluster))

        nparts = 100
        '''
        dStore = np.empty((len(XNocluster),len(XNocluster)))
        dStore[:] = np.nan
        for i in range(len(XNocluster)-1):
            for j in range(i+1,len(XNocluster)):
                dStore[i,j]=np.linalg.norm(XNocluster[i,:]-XNocluster[j,:],np.inf)
        #NOTE: ADDING NOISE INCREASES MEDIAN SUBSTANTIALLY
        # SINCE THE CLUSTERED POINTS HAVE A SURPRISINGLY SMALL SEPARATION
        #  COMPARED TO THE NOISE POINTS, WHICH HAVE NECESSARILY WIDER SPACEING                

        dStore = np.nanmin(dStore[:,1:],axis=0) #nearest neightbor distances
        dStore = dStore[dStore>0]
        d = np.nanmedian(dStore) #*2. #increase to let be convex
        print('Dmedina:',d)
        d = np.nanmean(dStore) #*2. #increase to let be convex
        print('Dmean:',d)
        dstd = np.nanstd(dStore) #*2. #increase to let be convex
        print('Dstd:',dstd)
        print('D_numroot',1./NUMPOINTS**(1/3.))
        d=d+3.*dstd;#seems ~~dnumroot for starlink AND navstar!
        print('D3std:',d)
        '''
        d=(1./NUMPOINTS)**(1/dims)



        nparts = int(np.ceil(1./d))
        #nparts=10
        #print('NPARTS:',nparts)
        XNocluster = np.copy(XNoclusterCopy)
        #quit()
        #nparts=10            
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

            #print('numreg: last %d, now %d'%(len(lastlistU),len(nowlistU)))
            runlayer = 1 if len(nowlistU)<len(lastlistU) else 0
            print('layertime', time.time() - start)

        print('alltime', time.time() - allstart)
        #quit()
            
        if injectScale>0:
            XNocluster = XNocluster[:injectLimit,:]
        print('XNocluster len: ',len(XNocluster))
        pt1list = []
        balllist = []
        radiuslist = []
        nfalist = []
        for node in nowlistU:

            node_tree = df_tree.loc[nowlist==int(node)]
            node_clusters = node_tree['cluster'].to_list()
            node_clusters = list(chain.from_iterable(node_clusters))

            ## REMOVE NOISE INJECTION
            node_clusters = np.asarray(node_clusters)
            #print('NODESTART:',len(node_clusters))
            if injectScale>0:
                node_clusters = node_clusters[node_clusters<injectLimit]
            #print('NODETRIM:',len(node_clusters))



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
            build_xyzplot(XNocluster,[], [1,1,1], folder=folder, name='%s_cluster%d'%(savename,counter),ptdetect = XCluster)

            #print('REPROJECTION DISABLED!')
            #runacclust = 0
        else:
            runacclust=0
        if len(XNocluster)<=1:
            runacclust=0

    #Data Association prior to outlier detection (for temporal aliasing)
    print('PREASSOC SIZE',templinesFilt.shape)
    if runoutlier==1:
        starfilter=4
        densityfilter=0.1
    else:
        starfilter=4
        densityfilter=0


    #Data Association prior to outlier detection (for temporal aliasing)
    templinesFilt,lines_full2,goodlinesfull2,numbeforefilter = associate_lines(goodidxfull,linessave,len(XFull),shape,postcluster,folder=folder,name=savename,spreadcorr=spreadcorr)
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
    CLUSTALLidx   = CLUST1idxfull | CLUST2idxfull | CLUST3idxfull | CLUSTXidxfull

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

    '''
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
    

    '''
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

