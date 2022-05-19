'''
PyCIS - Python Computational Inference from Structure

pylib/import_fits.py: Import and format FITS data and correct headers

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
Date of Modification: May 03, 2022

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
import glob
import numpy as np
from astropy.io import fits
import array
import imageio

def update_key(hdr,key):
    '''
    Update headers if comments are in value cells
    Inputs: hdr - FITS header object 
            key - string of header variable key
    Ouptuts: hdr - updated header object
    '''

    try:
        res = hdr[key].replace('.', '', 1).isdigit()
    except:
        res=False
    if not res:
        for i in hdr[key].split():
            try:
                temp=float(i)
                break
            except:
                continue
        try:
            del hdr[key]
            hdr[key] = temp
        except Exception as e:
            pass
    #print([key,hdr[key]])
    return hdr

def update_all_keys(hdr):
    '''
    Update headers if comments are in value cells
    Inputs: hdr - FITS header object 
    Ouptuts: hdr - updated header object
    '''
    #print(hdr['BZERO'])
    #print('BZERO1:',hdr['BZERO'])
    #hdr.set('BZERO', -32768.0, 'autofilled')
    #hdr.set('BSCALE', 1.0, 'autofilled')
    #print('BZERO1:',hdr['BZERO'])
    try:
        hdr=update_key(hdr,'BZERO')
    except:
        pass
    try:
        hdr=update_key(hdr,'BSCALE')
    except:
        pass
    try:
        hdr=update_key(hdr,'NAXIS1')
    except Exception as e:
        pass
    try:
        hdr=update_key(hdr,'NAXIS2')
    except:
        pass
    try:
        hdr=update_key(hdr,'CMND_RA')
    except Exception as e:
        pass
    try:
        hdr=update_key(hdr,'CMND_DEC')
    except:
        pass
    #print('BZERO2:',hdr['BZERO'])
    return hdr

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
        outfits='%s/%s'%(folder,infits.split("/")[-1])
        #open frame
        if len(headers)>(imnum-firstframe):
            print(outfits)
            data,hdrold=fits.getdata(infits,header=True)
            hdr = headers[imnum-firstframe]
            fits.writeto(outfits,data,hdr,overwrite=True, output_verify="ignore")

def import_fits(input_dir, savedata=0, subsample=1, framerange=[-1,-1], scale = 1, printheader=0,gauss=[1,1],median=0,shift=0,sig=0,binfactor=1,skip=False,
    bigmedian=0,bigmedianimg=0,minimg=np.inf,returnsize=False):
    '''
    Import fits files and either save or pass to main function 
    Returns images as a 3D numpy array and a list of FITS headers
    '''
    
    #specify pathing
    datatype='fit'

    #gather file names 
    #print([input_dir,datatype])
    imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
    listlen = len(imlist)
    if listlen==0:
        datatype='fits'
        imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
        listlen = len(imlist)
        if listlen==0: 
            #print('NO FIT OR FITS FILES...')
            #quit()
            datatype='fit.zip'
            imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
            listlen = len(imlist)
            if listlen==0: 
                print('NO FIT.ZIP FILES, CHANGE DATATYPE')
                quit()
    #print(imlist)
    fullsizeZ = np.copy(listlen)
    #filter input frame range for subsample
    firstframe = max(framerange[0],0)
    lastframe = min(framerange[1],listlen)
    if lastframe==-1:
        lastframe=listlen
    frames = []
    headers = []

    #Run video maker
    headerflag=0
    #print('Importing %d of %d ...'%(lastframe-firstframe+1,listlen))
    for imnum, infits in enumerate(imlist):
        #optional reduced frame subsampling
        if (subsample==1) and (imnum<firstframe or imnum>lastframe): 
            if not skip:
                pass#print('_%d_'%(imnum),end=', ',flush=True)
            continue 
        #open frame
        #printheader=1
        try:
            with fits.open(infits) as hdul:
                #print header and import data
                hdr = hdul[0].header
                hdr=update_all_keys(hdr)
                headers.append(hdr)

                headerflag=0; printheader=1;
                printheader=0;
                #printheader=1;
                if printheader==1 and headerflag==0:
                    headerflag=1
                    for entry in hdr:
                        print(entry,hdr[entry],hdr.comments[entry])
                    #print(hdr.info())
                    quit()
                #print('%d'%(imnum-firstframe),end=', ',flush=True)
                #print('%d'%(imnum),end=', ',flush=True)
                subs = hdul[0].data
                if skip:
                    subs=np.zeros((2,2))
                
                #format data for pycis
                subs = subs.astype(np.float64).squeeze()
                
                if subs.ndim==3: #TIME SERIES IN SINGLE FIGTS
                    #UNCOMMENT TO SHIFT DATA
                    a1 = int(((1.-scale) * subs.shape[1])/2.)
                    a2 = subs.shape[1] - a1
                    b1 = int(((1.-scale) * subs.shape[2])/2.)
                    b2 = subs.shape[2] - b1
                    frames=[]
                    headers=[]
                    #print('LOADING CUBE ')
                    #print(hdul.info())
                    for i in range(len(subs)):
                        #print header and import data
                        hdr = hdul[0].header
                        hdr=update_all_keys(hdr)
                        #headers.append(hdr)
                        #print('Trying header ',i)
                        #if 1==1:#printheader==1 and headerflag==0:
                        #    headerflag=1
                        #    for entry in hdr:
                        #        pass
                        #        #print(entry,hdr[entry],hdr.comments[entry])
                        #    #print(hdr.info())
                        #if i==len(subs)-1:
                        #    quit()
                        print('%d'%i,end=', ',flush=True)
                        ic=0#shift; #SET TO 2 TO SHIFT DATA 
                        ik = int(ic*i)
                        subsB = subs[i,a1:a2,b1-ik:b2-ik].squeeze()
                        if binfactor>1:
                            if (subsB.shape[0]%binfactor) > 0: 
                                subsB = subsB[:-int(subsB.shape[0]%binfactor),:]
                            if (subsB.shape[1]%binfactor) > 0: 
                                subsB = subsB[:,:-int(subsB.shape[1]%binfactor)]
                            nx = subsB.shape[0] // binfactor
                            ny = subsB.shape[1] // binfactor
                            subsB = subsB.reshape(nx,binfactor,ny,binfactor).sum(3).sum(1)
                        sub = subsB
                        frames.append(sub)
                        headers.append(hdr)
                    subs=np.asarray(frames)
                    frames=subs

                else: #TIME SERIES IN MULTIPLE FITS
                    #scale data for running on smaller machines
                    a1 = int(((1.-scale) * subs.shape[0])/2.)
                    a2 = subs.shape[0] - a1
                    b1 = int(((1.-scale) * subs.shape[1])/2.)
                    b2 = subs.shape[1] - b1
                    subsB = subs[a1:a2,b1:b2]
                    if binfactor>1:
                        if (subsB.shape[0]%binfactor) > 0: 
                            subsB = subsB[:-int(subsB.shape[0]%binfactor),:]
                        if (subsB.shape[1]%binfactor) > 0: 
                            subsB = subsB[:,:-int(subsB.shape[1]%binfactor)]
                        nx = subsB.shape[0] // binfactor
                        ny = subsB.shape[1] // binfactor
                        subsB = subsB.reshape(nx,binfactor,ny,binfactor).sum(3).sum(1)
                    frames.append(subsB)
        except Exception as e:
            print(e)
   
    frames = np.array(frames).squeeze()
    
    
    # Error capture for single frame
    if skip or (frames.ndim==2):
        #print('\nReturning single-frame image...')
        return frames, headers

    #Median-subtract if requested (off by default)
    if median==1:
        if bigmedian==0: #Compute the median over all frames
            meanframe = np.median(frames,axis=0)
        else: #Use the provided median image 
            meanframe = np.asarray(bigmedianimg)
        frames = frames - meanframe[np.newaxis,:,:]
    #minimg may be nan (compute for bigmedian solution and skip), inf (apply local min), or scalar 
    printminimg=0
    frames=np.asarray(frames)
    if np.isnan(minimg):#Find the median over all median-subtracted data
        printminimg=1
        minimg = np.amin(frames-np.median(frames,axis=0)[np.newaxis,:,:])
        print('NEWminimg:',minimg)
    elif np.isinf(minimg):#Set the minimum of each frame to 1
        frames = frames - np.amin(frames) +1
    else:#Use the provided minimum correction (ie, over all windows)
        frames = frames - minimg +1
    
    if not (shift==0):
        #print('SHIFT PROCESSING')
        ic=shift; #SET TO 2 TO SHIFT DATA
        maxshift = np.abs(int(len(frames)*ic))+1
        shiftframes = np.copy(frames[:,:,:-(maxshift)])
        for i in range(len(frames)):
            if shift>0:
                ik = int(ic*i)+1
                shiftframes[i,:,:] = frames[i,:,maxshift-ik:-ik].squeeze()
            else:
                ik = int(ic*i)-1
                shiftframes[i,:,:] = frames[i,:,-ik:-(maxshift+ik)].squeeze()
        frames = shiftframes

    if sig>0:
        #print('SIG PROCESSING')
        mean = np.mean(frames)
        div = np.std(frames)
        frames = np.clip(frames, 0, mean+sig*div)

    ## SAVE DATA
    if savedata==1:
        print('\nSaving image data...')
        np.save('%s/datacube.npy'%(input_dir),frames)
        print('Image data successfully saved!\n\n')
    else:
        #print('Returning image...',flush=True)
        if printminimg==1:
            return frames,headers,minimg
        if returnsize:
            fullsize = (frames.shape[1],frames.shape[2],fullsizeZ)
            #fullsize = (frames.shape[1],frames.shape[2],1)#fullsizeZ)
            return frames,headers,fullsize
        return frames, headers


def import_bin(input_dir, savedata=0, subsample=1, framerange=[-1,-1], scale = 1, printheader=0,median=1,shift=0):
    '''
    Import bin files and either save or pass to main function 
    Returns images as a 3D numpy array and a list of FITS headers
    '''
    #specify pathing
    datatype='bin'

    #gather file names 
    imname = '%s.%s'%(input_dir,datatype)
    if not os.path.exists(imname):
        print('FILE DOES NOT EXIST')
        quit() 
    print("IMPORTING ",imname)

    #imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
    #listlen = len(imlist)

    #filter input frame range for subsample
    f = open(imname,'rb')
    a = array.array("I") #uint64 header
    a.fromfile(f,1) #header is single integer size of file
    print(a)
    
    #get sizing data
    #dim = np.array([6200,6144,1])
    dim = np.array([6200,6144,1])
    print('LENG',len(a))
    nframes = int(a[0] / (np.prod(dim)*2)) #size of file divided by frame 2-byte pixels
    dim[2] = nframes
    listlen = nframes
    print('TOTAL FRAMES: ',nframes)

    #read pixels
    if not (a[-1] == np.prod(dim)*2):
        print('File data is not 6200 x 6144')
    else:
        a = array.array("H") #need uint16 loading
        a.fromfile(f, np.prod(dim))
        data = np.array(a).reshape(dim,order='F')
    f.close()
    #fix frame and scale
    firstframe = max(framerange[0],0)
    lastframe = min(framerange[1],listlen)
    if lastframe==-1:
        lastframe=listlen
    frames = []
    headers = []
   
    print('datashape',data.shape)
 
    for frame in range(firstframe,lastframe):
        subs = data[:,:,frame].squeeze()
        #format data for pycis
        subs = subs.astype(np.float64)
        #scale data for running on smaller machines
        a1 = int(((1.-scale) * subs.shape[0])/2.)
        a2 = subs.shape[0] - a1
        b1 = int(((1.-scale) * subs.shape[1])/2.)
        b2 = subs.shape[1] - b1
        subs = subs[a1:a2,b1:b2]
        frames.append(subs)
        headers.append([0])
            
    #save data
    frames = np.array(frames)
    print('FRAMESHAPE',frames.shape)
    print('stats: (mean,std,min,max) = ', (np.mean(frames), np.std(frames), frames.min(), frames.max()))
    
    #medain-subtract frame if requested (on by default)
    meanframe = np.median(frames,axis=0)
    if median==1:
        for frame in range(0,frames.shape[0]):
            frames[frame,:,:] = np.abs(frames[frame,:,:]- meanframe)
    frames = frames - np.amin(frames)+1

    ## SAVE DATA
    print(np.shape(np.array(frames)))
    if savedata==1:
        print('\nSaving image data...')
        np.save('%s/datacube.npy'%(input_dir),frames)
        print('Image data successfully saved!\n\n')
    else:
        print('\nReturning image...')
        return frames, headers


def import_png(input_dir, savedata=0, subsample=1, framerange=[-1,-1], scale = 1, printheader=0,gauss=[1,1],median=0,shift=0,sig=0,binfactor=1):
    '''
    Import fits files and either save or pass to main function 
    Returns images as a 3D numpy array and a list of FITS headers
    '''
    #specify pathing
    datatype='png'

    #gather file names 
    #print([input_dir,datatype])
    imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
    listlen = len(imlist)
    if listlen==0:
        datatype='jpg'
        imlist =  sorted(glob.glob('%s/*.%s'%(input_dir,datatype)))
        listlen = len(imlist)
        if listlen==0: 
            print('NO PNG OR JPG FILES, CHANGE DATATYPE')
            quit()
    #print(imlist)

    #filter input frame range for subsample
    firstframe = max(framerange[0],0)
    lastframe = min(framerange[1],listlen)
    if lastframe==-1:
        lastframe=listlen
    frames = []
    headers = []

    #Run video maker
    headerflag=0
    print('Importing %d of %d ...'%(lastframe-firstframe+1,listlen))
    for imnum, infits in enumerate(imlist):
        #optional reduced frame subsampling
        if (subsample==1) and (imnum<firstframe or imnum>lastframe): 
            continue 
        #open frame
        #printheader=1
        try:
            if 1==1:
                #print header and import data
                hdr = []
                headers.append(hdr)
                subs = imageio.imread(infits)
                
                #format data for pycis
                subs = subs.astype(np.float64).squeeze()
                if subs.ndim==3: #TIME SERIES IN SINGLE FIGTS
                    subs=subs[:,:,0].squeeze()
                    frames=subs
                a1 = int(((1.-scale) * subs.shape[0])/2.)
                a2 = subs.shape[0] - a1
                b1 = int(((1.-scale) * subs.shape[1])/2.)
                b2 = subs.shape[1] - b1
                #subs = subs[a1:a2,b1:b2]
                subsB = subs[a1:a2,b1:b2]
                if binfactor>1:
                    if (subsB.shape[0]%binfactor) > 0: 
                        subsB = subsB[:-int(subsB.shape[0]%binfactor),:]
                    if (subsB.shape[1]%binfactor) > 0: 
                        subsB = subsB[:,:-int(subsB.shape[1]%binfactor)]
                    nx = subsB.shape[0] // binfactor
                    ny = subsB.shape[1] // binfactor
                    subsB = subsB.reshape(nx,binfactor,ny,binfactor).sum(3).sum(1)
                frames.append(subsB)

        except Exception as e:
            print(e)
    ## PRINT STATS
    frames = np.array(frames).squeeze()
    #print('shape: ',np.shape(np.array(frames)))
    mean = np.mean(frames)
    dev = np.std(frames)
    #print('stats (mean,std,min,max): ', (mean, dev, frames.min(), frames.max()))
    mx =  frames.max()
    #print('max ',mx)
    
    # Error capture for single frame
    if frames.ndim==2:
        #print('\nReturning single-frame image...')
        return frames, headers

    #Median-subtract if requested (off by default)
    if median==1: 
        #print('MEDIAN PROCESSING')
        meanframe = np.median(frames,axis=0) 
        for frame in range(0,frames.shape[0]):
            frames[frame,:,:] = np.abs(frames[frame,:,:]- meanframe)
    #frames *=-1
    frames = frames - np.amin(frames)+1
    
    if not (shift==0):
        #print('SHIFT PROCESSING')
        ic=shift; #SET TO 2 TO SHIFT DATA 
        maxshift = np.abs(int(len(frames)*ic))+1
        shiftframes = np.copy(frames[:,:,:-(maxshift)])
        #print('shiftsize',shiftframes.shape)
        for i in range(len(frames)):
            #shiftframes[i,:,:] = frames[i,:,ik:ik-maxshift].squeeze()
            if shift>0:
                ik = int(ic*i)+1
                shiftframes[i,:,:] = frames[i,:,maxshift-ik:-ik].squeeze()
            else:
                ik = int(ic*i)-1
                #print([-ik,maxshift+ik])
                shiftframes[i,:,:] = frames[i,:,-ik:-(maxshift+ik)].squeeze()
        frames = shiftframes
    if sig>0:
        #print('SIG PROCESSING')
        mean = np.mean(frames)
        div = np.std(frames)
        #test1 = (mean-frames.min()) / div
        #test2 = (frames.max()-mean) / div
        #print('SIGRANGE: ',[test1,test2])
        #quit()
        frames = np.clip(frames, 0, mean+sig*div)
    mean = np.mean(frames)
    dev = np.std(frames)
    print('stats: ', (mean, dev, frames.min(), frames.max()))
    ### SAVE DATA
    if savedata==1:
        print('\nSaving image data...')
        np.save('%s/datacube.npy'%(input_dir),frames)
        print('Image data successfully saved!\n\n')
    else:
        #print('\nReturning image...')
        return frames, headers




