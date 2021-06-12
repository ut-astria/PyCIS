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
import glob
import numpy as np
from astropy.io import fits
#from astropy import convolution 
#from scipy.ndimage import gaussian_filter
#import sep

def mod_fits(input_dir, headers, folder='', subsample=1, framerange=[-1,-1]):
    '''    Update fits files with solved FITS headers    '''
    #specify pathing
    datatype='fit'

    #gather file names 
    imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
    listlen = len(imlist)

    #filter input frame range for subsample
    firstframe = max(framerange[0],0)
    lastframe = min(framerange[1],listlen)
    if lastframe==-1:
        lastframe=listlen

    #Update
    print('UPDATING DATA...')
    print('Updating %d of %d ...'%(lastframe-firstframe,listlen))
    if not os.path.exists('%s/'%folder):
        os.makedirs(folder)
    for imnum, infits in enumerate(imlist):
        #optional reduced frame subsampling
        if (subsample==1) and (imnum<firstframe or imnum>lastframe):
            continue
        #open frame
        if len(headers)>(imnum-firstframe):        
            outfits='%s/%s'%(folder, infits.split("/")[-1])
            #print(outfits)
            data, hdrold = fits.getdata(infits, header=True)
            hdr = headers[imnum-firstframe]
            fits.writeto(outfits,data,hdr,overwrite=True, output_verify="ignore")


def import_fits(input_dir, savedata=0, subsample=1, framerange=[-1,-1], scale = 1, printheader=0):
    '''
    Import fits files and either save or pass to main function 
    Returns images as a 3D numpy array and a list of FITS headers
    '''
    #specify pathing
    datatype='fit'

    #gather file names 
    imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
    listlen = len(imlist)

    #filter input frame range for subsample
    firstframe = max(framerange[0],0)
    lastframe = min(framerange[1],listlen)
    if lastframe==-1:
        lastframe=listlen
    frames = []
    headers = []

    #Run video maker
    headerflag=0
    print('IMPORTING DATA...')
    print('Printing %d of %d ...'%(lastframe-firstframe,listlen))
    for imnum, infits in enumerate(imlist):
        #optional reduced frame subsampling
        if (subsample==1) and (imnum<firstframe or imnum>lastframe): 
            continue 
        #open frame
        with fits.open(infits) as hdul:
            #print header and import data
            hdr = hdul[0].header
            headers.append(hdr)
            if printheader==1 and headerflag==0:
                headerflag=1
                for entry in hdr:
                    print(entry,hdr[entry],hdr.comments[entry])
                #print(hdr.info())

            print('%d'%(imnum-firstframe),end=', ',flush=True)
            subs = hdul[0].data
            

            #format data for pycis
            subs = subs.astype(np.float64)
            #scale data for running on smaller machines
            a1 = int(((1.-scale) * subs.shape[0])/2.)
            a2 = subs.shape[0] - a1
            b1 = int(((1.-scale) * subs.shape[1])/2.)
            b2 = subs.shape[1] - b1
            subs = subs[a1:a2,b1:b2]
            # (diabled) background subtraction 
            # subs = subs - sep.Background(subs)
            # subs = np.clip(subs,0,subs.max())
            #(disabled) normalize to 0/1 
            # subs = subs/np.max(subs)*subsval
            
            #append to image frame list 
            frames.append(subs)
            
    #save data
    frames = np.array(frames)
    print(np.shape(np.array(frames)))
    if savedata==1:
        print('\nSaving image data...')
        np.save('%s/datacube.npy'%(input_dir),frames)
        print('Image data successfully saved!\n\n')
    else:
        print('\nReturning image...')
        return frames, headers




