'''
PyCIS - Python Computational Inference from Structure

    A-contrario inference of object trajectories from structure-in-noise, 
    building on Line Segment Detection (LSD) for dense electro-optical time-series data
    formatted as 3D data cubes, with markov kernel estimation for non-uniform noise models.
    LSD C-extension module equipped with multi-layer a-contrario inference for center-line features
    from gradient information.  Python modules provided for inference of feature classifications
    using second-order gestalts, and ingesting/plotting of FITS-format data files.

    This software is under active development.

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
*Date of Modification: April 30, 2021

**NOTICE: For copyright and licensing, see 'notices' at bottom of README
'''
import os
import argparse
import numpy as np
import scipy as sp
from scipy import stats 
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
#Numpy Settings
np.set_printoptions(precision=2)
#np.seterr(all='ignore')

def demo_filt_obj(linesfull,demo=0):
    '''
    Compute a filter to remove the object of interest in the demo... 
    ''' 
    ldim = 4096*1
    l21=1025
    l22=1100
    l11=880
    l12=940
    lb=2068
    mysort = linesfull[:,0]>ldim*(l11/lb)
    mysort = mysort & (linesfull[:,0]<ldim*(l12/lb))
    mysort = mysort & (linesfull[:,1]>ldim*(l21/lb))
    mysort = mysort & (linesfull[:,1]<ldim*(l22/lb))
    mysort = mysort & (linesfull[:,3]>ldim*(l11/lb))
    mysort = mysort & (linesfull[:,3]<ldim*(l12/lb))
    mysort = mysort & (linesfull[:,4]>ldim*(l21/lb))
    mysort = mysort & (linesfull[:,4]<ldim*(l22/lb))
    sortlines = ~mysort
    if demo==0:
        sortlines = np.ones_like(sortlines)
    return sortlines


def RunBayes(k,xnum, X, wcp=1e-2,wtype='distribution',mpp=1e-2,cp=1):
    '''
    Compute GMM to rank r<k using variational bayesian inference
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
    elif numvars==4:
        # length-agnostic features (in case of fragmentation)
        #   (maxwidth, minwidth, az, el)
        linesNorm=linesNorm[:,[1,2,3,4]]
    else:
        #Simply select data
        linesNorm = linesNorm[:,:(numvars+1)]
    for x in range(linesNorm.shape[1]):
        if np.all(linesNorm[:,x]==linesNorm[0,x]):
            print("ERROR: detect_outliers: all elements on linesNorm index %d are identical"%x,flush=True)
            print("\t this may be due to, say, only detecting spatial features with elevation pi/2",flush=True)
            quit()
    #standardize data for pca
    linesNorm = standardize_data(linesNorm)
   
    #perform PCA decomposition
    covar = np.cov(linesNorm.T)
    evals,evecs = np.linalg.eig(covar)
    esort = np.argsort(-evals)
    evals = evals[esort]
    evecs = evecs[:,esort]
    
    #project data into PCA space
    proj = evecs[:,:]
    linesNormPCA = linesNorm.dot(proj)

    #return selected features, projected features, and projection
    return linesNorm, linesNormPCA, proj

def standardize_data(arr):
    '''
    https://towardsdatascience.com/pca-with-numpy-58917c1d0391
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    param 1: array 
    return: standardized array
    '''    
    rows, columns = arr.shape
    X=np.copy(arr) 
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        mean = np.mean(X[:,column])
        std = np.std(X[:,column])
        tempArray = np.empty(0)
        for element in X[:,column]:
            tempArray = np.append(tempArray, ((element - mean) / std))
        standardizedArray[:,column] = tempArray
    return standardizedArray



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

    #(Optionally) remove edge lines
    w=0#.1
    if w==0:
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


def detect_outliers(img,lines=[],folder='',savename='temp',args=None):
    '''
    Pipeline for NFA outlier detection 
    '''


    '''
    ------------------------------------------------------
    '''
    print('Filter Image and Line Data ')

    # Format image data 
    v = np.array(np.stack((img,)*3,axis=-1))
    aa,bb,cc,_ = v.shape
    aa=4096
    bb=4096
    cc=27

    #Format lines into proper matrix shape, and 
    #   (optionally) cut off spurious edge detections from gradient complications
    lines = format_lines(lines,aa=aa,filt=0)
    linessave=np.copy(lines)
    #Outlier detection only viable if there exist a 
    # statistically meaninful number of events 
    if len(linessave>100):

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
        el = np.arccos((lines[:,5]-lines[:,2])/k_len)
        az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
        # Correct angle ambiguity from (az[-pi,pi], el[-pi/2,pi/2]) to (az[-pi,pi], el[0,pi/2])
        # First perform rotation to correct elevation
        idx = el>np.pi/2
        el[idx] = np.pi - el[idx]
        az[idx] = az[idx]-np.pi
        # Then perform 2pi correction for azimuth 
        idx = az<(-1*np.pi)
        az[idx] = az[idx]+(2*np.pi)
        #rotate azimuth for visualization on cv2 images
        #  does not impact PCA separation 
        az = az - (np.pi/2)
        idx = az<(-1*np.pi)
        az[idx] = az[idx]+(2*np.pi)

        #Define the data to be used
        #   linkes[k,:] = [x,y,z,x,y,z,w,w,ptol,nfa]
        #   linesNorm: length, maxwidth, minwidth, az[-pi,pi], el[0,pi/2], nfa
        linesNorm = np.array([k_len,w1_sort,w2_sort,az,el,nfa]).T
            
        '''
        ------------------------------------------------------
        '''
        print('Set some variables')
        #Number of PCA components for decomposition
        pcanum=5 #length, az, el (width and nfa agnostic)
        #Number of PCA components for GMM separation
        xnum = max(pcanum-2,2)
        #GMM prior model setting 
        wtype='process'
        #Initial guess for number of GMM bases
        k=20
        #Specify demo to remove object from training set on demo file 
        demo = 1
        #Fraction of total data used for training GMM model
        pa = 0.75

        
        '''
        ------------------------------------------------------
        '''
        print('Separate Training/Testing Data')
        
        #Select lines which are not the object of interest
        sortOBJ = demo_filt_obj(np.copy(linessave),demo=demo)

        #Select a random pa% sample of the data 
        np.random.seed(1234) 
        sortCOUNT = np.random.choice(np.arange(2),size=len(linessave),p=[1.-pa,pa])

        #Select a training sample which is a 50% subsample excluding the object of interest
        sortFILT = sortCOUNT&sortOBJ
        linesFilt = np.copy(linesNorm)[sortFILT==1]

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


        '''
        ------------------------------------------------------
        '''
        print('Compute Gaussian Mixture Model ')
        '''
        https://towardsdatascience.com/gaussian-mixture-models-d13a5e915c8e
        GMM is an EM iteration, alternating gaussian paramters and soft assignment
        VariationalGMM(Bayesian) maximizes a lower bound on priors (is regularized)
        let N be number of lines and K be number of clusters
        BGMM gives R<K groups, cutting out low-probability terms
        however, it retains soft assignment probabilities to all K!
        proba:         (NxK) soft assignment probabilities, integrate to 1
        score_samples: (Nx1) likelihood of existence considering all K
        score:         (1x1) likelihood of observing all points considering all K
        we are given log-likelihood, and GMM minimizes *negative* log-likihood 
        '''

        #Run model on training data
        model = RunBayes(np.copy(k),np.copy(xnum),np.copy(XFilt),wtype=np.copy(wtype))
        #Apply model to testing data 
        X = np.copy(XFull)
        labels = model.predict(X) #Assignment to clusters
        #individual probabilities for each gaussian
        predict_proba = model.predict_proba(X) 
        #(instantiate) negative log10 likelihood of existance (will be overwritten)
        #TODO: confirm interpretation, why we don't just use this value directly 
        score_samples = -model.score_samples(X) / np.log(10)
        



        '''
        ------------------------------------------------------
        '''
        print('Rank r<k Reduction and Log-Liklihood Computation')

        #define gmm-weighted probabilities 
        predict_proba2 = np.copy(predict_proba)*model.weights_

        #identify indices for rank r<k data
        numdist = len(np.unique(labels))
        print('Rank: r=%d , k=%d '%(numdist,len(model.weights_)))
        weightsort = np.array(model.weights_).argsort()[-numdist:][::-1]

        #select rank r<k data 
        predict_proba2 = predict_proba2[:,weightsort]
        weights2 = model.weights_[weightsort]
        means2 = model.means_[weightsort]
        covar2 = model.covariances_[weightsort]
        #precompute some constants to speed up further iteration  
        covarinv = np.linalg.inv(covar2)
        pnorms = np.linalg.norm(predict_proba2,axis=1)

        #precompute log-likelihoods for all k weights 
        #dist_all = np.array([np.log(model.weights_[u])+
        #    stats.multivariate_normal.logpdf(x=X,mean=model.means_[u],cov=model.covariances_[u]) for u in range(len(model.weights_))]).T
    
        '''
        ------------------------------------------------------
        '''
        print('Mahanalobis Analysis')


        #initialize memory 
        mahanrow = np.copy(predict_proba[0])[:numdist]
        mahan = np.copy(score_samples)

        #define prior model for scaled-mahanobis-distance distribution 
        fn,fm = X.shape
        fc = (fn-fm)/((fn-1.)*fm)
        frv = stats.f(fm,fn-fm,scale=1.)

        #for each line sample...
        method = 'max'
        for i in range(len(score_samples)):
            #select the local r<k data for computations
            #dist = dist_all[i] 
            tempmahan = np.copy(mahanrow)
            pnorm = pnorms[i]
            #compute mahanalobis distance to cluster U
            #means2/covarinv are sorted according to descending weight, to rank r<k
            for u in range(numdist):#range(len(model.weights_)):
                mahanrow[u] = ((X[i]-means2[u]).T @  covarinv[u] @ (X[i]-means2[u]))
            #the *scaled* mahanalobis distances have a known distribution 
            mahanrow = fc * mahanrow 
            
            #Compute likelihood of existence - maximal or joint method
            #We seek the smallest probability, or largest negative log-likelihood
            if method=='max':
                #Find largest likelihood of belonging to any set
                #Get P(X>x) p-values 
                mahanrow = frv.cdf(mahanrow) 
                tempmahan = 1.-mahanrow
                #Select maximum p-value 
                mahan[i] = max(tempmahan)
            elif mathod=='sum':
                #Find joint likelihood of belonging to any set 
                #Get P(X<=x) : 1-"p-value" 
                mahanrow = frv.cdf(mahanrow) 
                #Convert P values to z-score
                mahanrow = stats.norm.ppf(mahanrow)
                #Weight z scores using gmm-weighted probabilites
                mahanrow = predict_proba2[i]*mahanrow
                #Compute p-value of weighted z-score sum (ensure normalization)
                mahan[i] = 1. - stats.norm.cdf(np.sum(mahanrow)/pnorm)

        #Convert existance probabilities to negative-log-likelihoods
        mahan = -np.log10(mahan)
        #Multiply by the number of total possible successes (yeilds NFA)
        N1 = -np.log10(len(mahan)) #-np.log10(len(X))
        score_samples = mahan + N1 

        '''
        ------------------------------------------------------
        '''
        print('Applying NFA Principle')

        #Define NFA threshold (as -log10(e)) 
        eps=0 #-log10(nfa)<-log10(e=1)
        print('min: %.2f, max: %.2f, eps: %.2f'%(min(score_samples),max(score_samples),eps))
        
        #Compare NFA to threshold 
        goodidxfull = score_samples>eps
        badidxfull = score_samples<=eps
       
    else:
        print('Cannot perform outlier detection on only %d lines'%len(linessave))
        goodidxfull = np.ones((len(linessave),))==1
        badidxfull = np.zeros((len(linessave),))==1

    lines_full = np.copy(linessave)
    goodlinesfull = lines_full[goodidxfull]
    badlinesfull = lines_full[badidxfull]
    print('Total lines: %d, Accepted: %d, Rejected: %d'%(len(lines_full), 
        np.count_nonzero(goodidxfull),np.count_nonzero(badidxfull) ))
    
    np.save('%s/goodlines_%s.npy'%(folder,savename),goodlinesfull)
    np.save('%s/badlines_%s.npy'%(folder,savename),badlinesfull)

    print('Done!\n\n')
    return goodlinesfull, badlinesfull




if __name__=='__main__':
    print('-------MAKING VIDS-------\n\n')
    parser = argparse.ArgumentParser(description='ASTRIA Telescope Image Pipeline')
    parser.add_argument("-n","--name",type=str,help='base name of image and line targets') 
    parser.add_argument("-fa","--framea",type=int,default=8,help='first frame of video') 
    parser.add_argument("-fb","--frameb",type=int,default=33,help='second frame of video') 
    parser.add_argument("-s","--scale",type=float,default=1.,help='scale of video') 
    args = parser.parse_args()


     ## LOAD IMAGES 
    satfolder='data'
    outfolder='results'
    ## IF NOT ON TACC, USE THE SETTINGS IN FOLLOWING COMMENTS
    framerange=[args.fa,args.fb]
    scale = args.s

    #import image data and reshape as needed
    satname = args.n
    prename='{}'.format(satname)
    input_dir = '%s/%s'%(satfolder,satname)
    I3 = import_fits(input_dir, savedata=0, subsample=1, 
        framerange=framerange, scale=scale)
    I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))

    #import line data
    if os.paty.isfile('%s/data2_%s'%(outfolder,args.n)):
        print('Using center lines')
        lines=np.load('%s/data2_%s'%(outfolder,args.n))
    elif os.paty.isfile('%s/data1_%s'%(outfolder,args.n)):
        print('Using edge lines')
        lines=np.load('%s/data1_%s'%(outfolder,args.n))
    else:
        print('ERROR: No data provided for name')
        quit()

    detect_outliers(np.copy(I3),lines,folder=outfolder,savename=args.n,args=args)



