'''
PyCIS - Python Computational Inference from Structure

pylib/print_detections.py: Generate still and animated visuals of detections, star/noise lines, and clustering details.   

TODO:
  Link html output to online ASTRIANet visualization 

Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
Date of Modification: May 5, 2022

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
import pandas as pd
from scipy.ndimage import gaussian_filter
import imageio
import cv2 
import plotly.graph_objects as go
from pylib.import_fits import import_fits

def make_kernel(medlen,medwid,medaz,subtract=False):
    '''Build a convolution kernel given a set of length/width/orientation data'''
    #sepkernY = int(np.ceil(medlen*np.cos(medaz)))
    #sepkernX = int(np.ceil(medlen*np.sin(medaz)))
    sepkernX = int(np.ceil(medlen*np.cos(medaz)))
    sepkernY = int(np.ceil(medlen*np.sin(medaz)))


    medlen = int(medlen)
    sepkern = np.zeros((5*int(medlen),5*int(medlen)))
    medlen = int(np.ceil(2.5*medlen))
    sepkern = cv2.line(sepkern,(medlen,medlen),(medlen+sepkernX,medlen+sepkernY),1,int(np.ceil(medwid)))
    sepkern = gaussian_filter(sepkern,sigma=1,mode='constant',cval=0)
    sepkern = np.ceil(sepkern*8) #the usual 1-3-5-8 filter for a 3x3 circular image
    mask0 = np.amax(sepkern,axis=1)>0.
    sepkern = sepkern[mask0,:]
    mask1 = np.amax(sepkern,axis=0)>0.
    sepkern = sepkern[:,mask1]
    #print('kernel shape ',sepkern.shape)
    if (sepkern.shape[0]%2)==0:
        sepkern = sepkern[:-1,:]
    if (sepkern.shape[1]%2)==0:
        sepkern = sepkern[:,:-1]

    if subtract:
        diff=np.abs(sepkern.shape[0]-sepkern.shape[1])
        diff=int(np.floor(diff/2.))
        sepkern[sepkern<1] = 0
        if sepkern.shape[0]>sepkern.shape[1]:
            sepkern = np.pad(sepkern,((0,0),(diff,diff)),constant_values=0)
        elif sepkern.shape[0]<sepkern.shape[1]:
            sepkern = np.pad(sepkern,((diff,diff),(0,0)),constant_values=0)
        zerocount = sepkern.size - np.count_nonzero(sepkern)
        sepkern[sepkern==0] = -1. * np.sum(sepkern) / float(zerocount) #net sum is zero
    
    #sepkern = sepkern / np.sum(sepkern) #normalize for safety
    return sepkern

def prepimg(img,amp,val_max=np.nan):
    '''    Scale image for printing purposes     '''
    v = np.copy(img)
    print('printvalmax:',val_max)
    #NOTE: img is usually background-subtracted [0,(2^16)-1] uint16
    #Printing is a 3-sigma clipping scaled to [0,255] uint8
    if amp==1:
        if np.isnan(val_max):
            val_max = np.mean(v) + 5*np.std(v)
        else:
            pass
        means = v.mean(axis=(0,1))
        stds = v.std(axis=(0,1))
        
        for f in range(v.shape[2]):
            vf = np.copy(v[:,:,f])
            vmax = min(means[f] + 5*stds[f],65535)#min(np.mean(vf)+5*np.std(vf),65535)
            vmin = max(means[f] - 5*stds[f],0)#max(0,np.mean(vf)-5*np.std(vf))
            vf = np.clip(vf,vmin,vmax)
            vf = (vf-vmin)/(vmax-vmin)
            vf = (vf/np.max(vf))*255
            v[:,:,f] = np.copy(vf)
        v = v.astype(np.uint8)

    else:
        v = np.array((v-v.min())/(v.max()-v.min())*255,np.uint8)
        v = np.array((v-v.min())/(v.max()-v.min())*255,np.uint8)
    v = np.stack((v,)*3,axis=-1)
    v = np.array(v)
    return v


def myinterp(x,x1,x2,y1,y2,extrap=False):
    '''     let x be the z index and 'y' be x or y  '''
    if (((x1>x2) or (x<x1)) or (x>x2)) and (not extrap):
        #print('error: intep: x1>x2 or x<x1 or x>x2')
        return None
    xeq = np.abs(x1-x2)<(1e-8)
    if xeq and (y1<y2):
        return y1
    if xeq and (y2<y1):
        return y2
    try:
        return y1 + (x-x1)*(y2-y1)/(x2-x1)
    except:
        return min(y1,y2)


def interp_frame(locline,z,double=False,extrap=False,printonly=False):
    '''    Get x,y line to draw on z-level of video     '''
    if double:
        x1 = float(locline[1]); y1 = float(locline[0]); z1 = float(locline[2])
        x2 =float(locline[4]); y2 = float(locline[3]); z2 = float(locline[5])
        z=float(z)
    else:
        x1 = int(locline[1]); y1 = int(locline[0]); z1 = int(locline[2]);
        x2 = int(locline[4]); y2 = int(locline[3]);z2 = int(locline[5]);
        z=int(z)
    if  (z<min(z1,z2)) and (not extrap):
        return 0,0,0,0
    if (z>max(z1,z2)) and (not extrap):
        if printonly:
            return float(x1),float(y1),float(x2),float(y2)
        else:            
            return 0,0,0,0
    if z1>z2:
        tempx = np.copy(x1); tempy = np.copy(y1); tempz = np.copy(z1)
        x1 = np.copy(x2); y1 = np.copy(y2); z1 = np.copy(z2)
        x2 = tempx; y2 = tempy; z2 = tempz
    if True:#(z<z2) and (z1!=z2):
        x2 = myinterp(z,z1,z2,x1,x2,extrap)
        y2 = myinterp(z,z1,z2,y1,y2,extrap)
        if (x2 is not None) and (y2 is not None):
            if double:
                x2 = float(x2); y2 = float(y2)
            else:
                x2 = int(x2); y2 = int(y2)
        else:
            return 0,0,0,0
    #convert to float required for scaling process, will be cast back to int
    #x1+=1;y1+=1;x2+=1;y2+=1;
    return float(x1),float(y1),float(x2),float(y2)

def interp_frame_Z(locline,z):
    '''    Get x,y line to draw on z-level of video     '''
    x1 = int(locline[1]); y1 = int(locline[0]); z1 = int(locline[2]);
    x2 = int(locline[4]); y2 = int(locline[3]); z2 = int(locline[5]);
    if (z<min(z1,z2)):
        return 0,0,0,0,0
    if z1>z2:
        tempx = np.copy(x1); tempy = np.copy(y1); tempz = np.copy(z1)
        x1 = np.copy(x2); y1 = np.copy(y2); z1 = np.copy(z2)
        x2 = tempx; y2 = tempy; z2 = tempz
    if (np.ceil(z)<np.ceil(z2)) and (z1!=z2):
        x2 = myinterp(z,z1,z2,x1,x2)
        y2 = myinterp(z,z1,z2,y1,y2)
        if (x2 is not None) and (y2 is not None):
            x2 = int(x2); y2 = int(y2)
        else:
            return 0,0,0,0,0
    #convert to float required for scaling process, will be cast back to int
    #x1+=1;y1+=1;x2+=1;y2+=1;
    return float(x1),float(y1),float(x2),float(y2),int(z2)

def interp_frame_xy(locline,z, star=0,double=False,extrap=False,shape=[],printdetail=False,printonly=False):
    ''' Get x,y line to draw on z-level of video, accound for single-frame streaks '''
    if double:
        z1 = float(locline[2]); z2 = float(locline[5]);
        z=float(z);
    else:
        z1 = int(locline[2]); z2 = int(locline[5]);
        z=int(z);

    if not extrap:
        if (z<min(z1,z2)) or (z>max(z1,z2)): #(np.ceil(z)>np.ceil(max(z1,z2))) or (np.floor(z)<np.floor(min(z1,z2))):
            if printdetail:
                pass#print('\t\t z %d out of Z bounds: %.2f [%.2f,%.2f]'%(z,z,z1,z2))
            return 0,0

    x1,y1,x2,y2 = interp_frame(locline,z,double=double,extrap=extrap,printonly=printonly)
   
    if (z>max(z1,z2)) and (not extrap):
        if printdetail:
            pass#print('\t\t z %d out of bounds II'%z)
        return 0,0
    if (star==1) and not ((np.abs(z1-z2)<(1e-8)) and (np.abs(z1-z)<(1e-8))):
        if printdetail:
            pass#rint('\t\t z %d first star catch'%z)
        return 0,0
    if (star>1) and not (np.abs(z1-z2)<(star-1)):
        if printdetail:
            pass#print('\t\t z %d second star catch'%z)
        return 0,0
    if np.abs(z1-z2)<(1e-8):
      x = (x1+x2)/2.; y = (y1+y2)/2.;
    else:
      x = x2; y = y2;
    if len(shape)>0:
        if (((x<0.) | (x>shape[0])) | ((y<0.) | (y>shape[1]))) | ((z<0.) | (z>shape[2])):
            if printdetail:
              pass#print('\t\t z %d out of XYZ bounds'%z)
            return 0,0
        else:
            pass
    return x,y

def interp_frame_draw(locline,z, vs,printonly=False):
    ''' Get x,y line to draw on z-level of video, accound for single-frame streaks '''
    x1,y1,x2,y2 = interp_frame(locline,z,printonly=printonly)
    z1 = int(locline[2]); z2 = int(locline[5]);
    if (z>max(z1,z2)):
        return 0,0,0,0
    else:
        x1*=vs; y1*=vs; x2*=vs; y2*=vs
        x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
        return x1,y1,x2,y2

def print_detections(img,goodlines=[],badlines=[],folder='',savename='temp',args=None,makeimg=1,makevid=0,makegif=1,vs=1,fps=5,amp=0,background=0):#vs=0.25):i
    '''    Generate video    '''
    print('%d 2nd-order meaningful lines, %d other 1st-order meaninful lines'%(len(goodlines),len(badlines)))

    #Prepare image data for printing
    v = prepimg(img,amp)
    aa,bb,cc,_ = v.shape
    av=int(bb*vs); bv=int(aa*vs); cv=int(cc*vs)
    gifset_all=[]
    gifset_obj=[]

    #Get a line thickness for printing
    denom=8
    thick = int(np.ceil(v.shape[0]/100)/denom)
    if vs<1.:
        thick = int(thick*max(vs,0.5))
    thick = max(thick,2)
    #thick=1
    ## Initialize video writer
    #fps = 5.0
    if not os.path.exists(folder):
        os.makedirs(folder)
    videoname_all='%s/videoAll_%s.avi'%(folder,savename)
    videoname_obj='%s/videoObj_%s.avi'%(folder,savename)
    if makevid==1:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_all = cv2.VideoWriter(videoname_all,fourcc,fps,(av,bv))
        video_obj = cv2.VideoWriter(videoname_obj,fourcc,fps,(av,bv))

    #Run video maker
    print('writing : ',savename, '...')
    sub_all = np.copy(v[:,:,-1,:].squeeze())
    sub_all = cv2.resize(sub_all, (av,bv), interpolation = cv2.INTER_CUBIC)
    sub_all = cv2.convertScaleAbs(sub_all)
    sub_img = np.ones_like(sub_all)*int(255/2)
    if background:
        sub_img = np.copy(sub_all)
    viridis_seedB = np.array([range(1,255,int(np.ceil(255/cc)))]).astype(np.uint8)
    viridis_seed = np.array([np.linspace(1,255,num=int(cc))]).astype(np.uint8)
    #print(viridis_seedB)
    #print(viridis_seed)
    print('recording of stars on still image disabled around line 204 (subimg')
    viridis_list = cv2.applyColorMap(viridis_seed,cv2.COLORMAP_VIRIDIS)[0]
    for z in range(int(cc)):
        tcolor = float(z)/float(cc)
        #Get frame
        
        #Create temp data frames
        sub_all = np.copy(v[:,:,z,:].squeeze())
        sub_all = cv2.resize(sub_all, (av,bv), interpolation = cv2.INTER_CUBIC)
        sub_all = cv2.convertScaleAbs(sub_all)
        sub_obj = np.copy(sub_all)
        subwrite_all = np.copy(sub_all)
        subwrite_obj = np.copy(sub_all)

        #Interpolate (bad)line to draw at z-frame
        
        if len(badlines)>0:
            for k in range(len(badlines)):
                locline = badlines[k,:6].squeeze()
                x1,y1,x2,y2,z2 = interp_frame_Z(locline,z)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                z2=min(z2,int(cc)-1)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                #sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),(255,0,0),thick)
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),viridis_list[int(z2)].tolist(),thick)
                x1,y1,x2,y2 = interp_frame_draw(locline,z,vs,printonly=True)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                #sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),(int(255*tcolor),int(255*(1.-tcolor)),0),thick)
                #sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),viridis_list[int(z)].tolist(),thick)

        #Interpolate (good)line to draw at z-frame OVER bad frames
        # so that good frames are not obstructed from visibility
        if len(goodlines)>0:
            for k in range(len(goodlines)):
                locline = goodlines[k,:6].squeeze()
                x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                #if len(badlines)>0:
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),(0,0,255),thick)
                sub_obj = cv2.line(sub_obj,(x1,y1),(x2,y2),(0,0,255),thick)
                x1,y1,x2,y2 = interp_frame_draw(locline,z,vs,printonly=True)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                #sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),(int(255*(1-tcolor)),0,int(255*tcolor)),thick)
                sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),(int(255*(0)),0,int(255)),thick)

        #Draw frames with transparency, to ensure visibility of all lines
        alpha = 0.5
        if alpha<1.:
            cv2.addWeighted(sub_all,alpha,subwrite_all,1-alpha,0,subwrite_all)
            cv2.addWeighted(sub_obj,alpha,subwrite_obj,1-alpha,0,subwrite_obj)
        else:
            subwrite_all=sub_all
            subwrite_obj=sub_obj
        #added for possible disabling of vido printing
        if makevid==1:
            video_all.write(subwrite_all)
            video_obj.write(subwrite_obj)
            #write gif data 
        if makegif==1:
            subwrite_all = cv2.cvtColor(subwrite_all, cv2.COLOR_BGR2RGB)
            subwrite_obj = cv2.cvtColor(subwrite_obj, cv2.COLOR_BGR2RGB)
            gifset_all.append(subwrite_all)
            gifset_obj.append(subwrite_obj)

    #release video
    print('releasing...')
    if makevid==1:
        video_all.release()
        video_obj.release()
    if makegif==1:
        #save reduced-scale gifs
        pre, ext = os.path.splitext(videoname_all)
        gifname_all = pre+".gif"
        pre, ext = os.path.splitext(videoname_obj)
        gifname_obj = pre+".gif"
        imageio.mimsave(gifname_all, gifset_all,fps=fps)
        imageio.mimsave(gifname_obj, gifset_obj,fps=fps)
        #imageio.mimsave(gifname_obj, gifset_obj,fps=5)

    #Saving last frame for visualization 
    imageio.imwrite('%s/img_%s.png'%(folder,savename),np.flip(sub_img,-1))
    #imageio.imwrite('%s/img_%s.png'%(folder,savename),np.flip(sub_all[-300:,:300,:],-1))

    print('Done!\n\n')


def print_detections_clust(img,loadfolder='',loadname='',folder='',savename='temp',args=None,makeimg=1,makevid=0,makegif=1,vs=1,fps=5,amp=0):#vs=0.25):
    '''    Generate video    '''
    #print('%d 2nd-order meaningful lines, %d other 1st-order meaninful lines'%(len(goodlines),len(badlines)))


    CLUST1 = np.load('%s/CLUST1_%s.npy'%(loadfolder,loadname))
    CLUST2 = np.load('%s/CLUST2_%s.npy'%(loadfolder,loadname))
    CLUST3 = np.load('%s/CLUST3_%s.npy'%(loadfolder,loadname))
    CLUSTX = np.load('%s/CLUSTX_%s.npy'%(loadfolder,loadname))
    REMAIN = np.load('%s/REMAIN_%s.npy'%(loadfolder,loadname))
    OUTLIER = np.load('%s/OUTLIER_%s.npy'%(loadfolder,loadname))

    print('CLUST LEN:')
    print('1: ',len(CLUST1))
    print('2: ',len(CLUST2))
    print('3: ',len(CLUST3))
    print('X: ',len(CLUSTX))
    print('R: ',len(REMAIN))
    print('O: ',len(OUTLIER))

    #Prepare image data for printing
    v = prepimg(img,amp)
    aa,bb,cc,_ = v.shape
    av=int(bb*vs); bv=int(aa*vs); cv=int(cc*vs)
    gifset_all=[]

    #Get a line thickness for printing
    denom=8
    thick = int(np.ceil(v.shape[0]/100)/denom)
    if vs<1.:
        thick = int(thick*max(vs,0.5))
    thick = max(thick,2)
    #thick=1
    ## Initialize video writer
    #fps = 5.0
    if not os.path.exists(folder):
        os.makedirs(folder)
    videoname_all='%s/videoAllClust_%s.avi'%(folder,savename)
    if makevid==1:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_all = cv2.VideoWriter(videoname_all,fourcc,fps,(av,bv))

    #Run video maker
    print('writing : ',savename, '...')
    sub_all = np.copy(v[:,:,0,:].squeeze())
    sub_all = cv2.resize(sub_all, (av,bv), interpolation = cv2.INTER_CUBIC)
    sub_all = cv2.convertScaleAbs(sub_all)
    sub_img = np.ones_like(sub_all)*int(255/2)
    viridis_seed = np.array([np.linspace(1,255,num=5)]).astype(np.uint8)
 
    viridis_list = cv2.applyColorMap(viridis_seed,cv2.COLORMAP_VIRIDIS)[0]

    for z in range(int(cc)):
        #Get frame
        
        #Create temp data frames
        sub_all = np.copy(v[:,:,z,:].squeeze())
        sub_all = cv2.resize(sub_all, (av,bv), interpolation = cv2.INTER_CUBIC)
        sub_all = cv2.convertScaleAbs(sub_all)
        subwrite_all = np.copy(sub_all)

        #Interpolate (bad)line to draw at z-frame
        templines = CLUSTX
        if len(templines)>0:
            for k in range(len(templines)):
                locline = templines[k,:6].squeeze()
                x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),viridis_list[1].tolist(),thick)
                x1,y1,x2,y2 = interp_frame_draw(locline,z,vs)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),
                    viridis_list[1].tolist(),thick)
        templines = CLUST3
        if len(templines)>0:
            for k in range(len(templines)):
                locline = templines[k,:6].squeeze()
                x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),viridis_list[2].tolist(),thick)
                x1,y1,x2,y2 = interp_frame_draw(locline,z,vs,printonly=True)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),
                    viridis_list[2].tolist(),thick)
        templines = CLUST2
        if len(templines)>0:
            for k in range(len(templines)):
                locline = templines[k,:6].squeeze()
                x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),viridis_list[3].tolist(),thick)
                x1,y1,x2,y2 = interp_frame_draw(locline,z,vs,printonly=True)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),
                    viridis_list[3].tolist(),thick)
        templines = CLUST1
        if len(templines)>0:
            for k in range(len(templines)):
                locline = templines[k,:6].squeeze()
                x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),viridis_list[4].tolist(),thick)
                x1,y1,x2,y2 = interp_frame_draw(locline,z,vs,printonly=True)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),
                    viridis_list[4].tolist(),thick)
        templines = REMAIN
        if len(templines)>0:
            for k in range(len(templines)):
                locline = templines[k,:6].squeeze()
                x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),viridis_list[0].tolist(),thick)
                x1,y1,x2,y2 = interp_frame_draw(locline,z,vs,printonly=True)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),
                    viridis_list[0].tolist(),thick)
        templines = OUTLIER
        if len(templines)>0:
            for k in range(len(templines)):
                locline = templines[k,:6].squeeze()
                x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),(0,0,255),thick)
                x1,y1,x2,y2 = interp_frame_draw(locline,z,vs,printonly=True)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),
                    (0,0,255),thick)

        #Draw frames with transparency, to ensure visibility of all lines
        alpha = 0.5
        if alpha<1.:
            cv2.addWeighted(sub_all,alpha,subwrite_all,1-alpha,0,subwrite_all)
        else:
            subwrite_all=sub_all
        #added for possible disabling of vido printing
        if makevid==1:
            video_all.write(subwrite_all)
            #write gif data 
        if makegif==1:
            subwrite_all = cv2.cvtColor(subwrite_all, cv2.COLOR_BGR2RGB)
            gifset_all.append(subwrite_all)
    #release video
    print('releasing...')
    if makevid==1:
        video_all.release()
    if makegif==1:
        #save reduced-scale gifs
        pre, ext = os.path.splitext(videoname_all)
        gifname_all = pre+".gif"
        imageio.mimsave(gifname_all, gifset_all,fps=fps)

    #Saving last frame for visualization 
    imageio.imwrite('%s/imgclust_%s.png'%(folder,savename),np.flip(sub_img,-1))
    tester = np.zeros((500,100,3))
    for i in range(5):
        tester[int(i*100):int((i+1)*100),:,:] = np.broadcast_to(viridis_list[i],(100,100,3))
    imageio.imwrite('%s/legend.png'%folder,np.flip(tester,-1))
    print('Done!\n\n')

def build_xyzplot(xyzarray,lines,shape,folder='.',name='association_all.html',volume=[],ptdetect=[],rdt=[],rdtidx=[],title=None,vscale=1):

    xname='x_image' #'xspace'
    yname='y_image' #'xspace'
    zname='time_frame' #'zframe'
    vname='value'
    X,Y,Z = shape
    figlist=[]
    if len(lines)>0:
        for k in range(len(lines)):
            locline = lines[k,:6].squeeze()
            #print(locline)
            locline = np.reshape(locline,(2,3))
            df = pd.DataFrame(locline, columns=[xname,yname,zname])
            #figlist.append(px.line_3d(df,x=xname,y=yname,z=zname))
            if len(rdt)==0:
                fig = go.Figure(data=[go.Scatter3d(x=df[xname],y=df[yname],z=df[zname],mode='lines',showlegend=False,hoverinfo="skip",
                line=dict(
                    width=4,color='rgba(255,0,0,0.5)'
                    ))])
            else:
                fig = go.Figure(data=[go.Scatter3d(x=df[xname],y=df[yname],z=df[zname],mode='lines',showlegend=False,hovertext="Detection from A-Contrario pipeline, with filtering",hoverinfo="text",
                line=dict(
                    width=6,color='rgba(0,255,0,0.25)'
                    ))])
            figlist.append(fig)

    if ((len(xyzarray)>0) and (len(volume)==0)):
        print("build_xyzplot: considering point input")
        df = pd.DataFrame(xyzarray,columns=[xname,yname,zname])
        if len(ptdetect)>0:
            print("\t considering classified-population point input")

            fig = go.Figure(data=[go.Scatter3d(x=df[xname],y=df[yname],z=df[zname],mode='markers',showlegend=False,hoverinfo="skip",
                marker=dict(
                    size=5,color='rgba(0,0,255,0.25)'
                    ))])
            figlist.append(fig)
            df = pd.DataFrame(ptdetect,columns=[xname,yname,zname])
            fig = go.Figure(data=[go.Scatter3d(x=df[xname],y=df[yname],z=df[zname],mode='markers',showlegend=False,hoverinfo="skip",
                marker=dict(
                    size=5,color='rgba(255,0,0,0.25)'
                    ))])
            figlist.append(fig)
        else:
            print("\t considering single-population point input")
            fig = go.Figure(data=[go.Scatter3d(x=df[xname],y=df[yname],z=df[zname],mode='markers',showlegend=False,hoverinfo="skip",
                marker=dict(
                    size=12,color=df[zname],colorscale='Viridis',opacity=0.5
                    ))])
            figlist.append(fig)
    if len(rdt)>0:
        print('xylen %d, rdtlen %d, xyzrdttrue %d'%(len(xyzarray),len(rdt),np.count_nonzero(rdtidx)))
        df = pd.DataFrame(xyzarray[rdtidx,:],columns=[xname,yname,zname])
        print("\t considering single-population point input")
        hovertext = []
        for r in rdt:
            hovertext.append("RA %s dms<br>DEC %s dms<br>UTC %s"%(r[0],r[1],r[2]))

        fig = go.Figure(data=[go.Scatter3d(x=df[xname],y=df[yname],z=df[zname],mode='markers',showlegend=False,hovertext=hovertext,hoverinfo="text",#hoverinfo="skip",
            marker=dict(
                size=5,color='rgba(255,0,0,0.5)'
                ))])
        figlist.append(fig)

        df = pd.DataFrame(xyzarray[~rdtidx,:],columns=[xname,yname,zname])
        print("\t considering single-population point input")
       

        fig = go.Figure(data=[go.Scatter3d(x=df[xname],y=df[yname],z=df[zname],mode='markers',showlegend=False,hovertext="Astrometric detection rejected - unable to find source near A-Contrario trajectory result",hoverinfo="text",
            marker=dict(
                size=5,color='rgba(0,0,255,0.5)'
                ))])
        figlist.append(fig)

   


    #start figure
    fig = go.Figure()
    # Add data to be displayed before animation starts
    datalist = [fig.data[0] for fig in figlist]
    for datael in datalist:
        fig.add_trace(datael)
    # Now check for volume
    if len(volume)>0:
        #xyzarray=np.ceil(xyzarray).astype(np.uint16)
        xyzarray=np.asarray(xyzarray).astype(float)
        #print(xyzarray)
        #prepare volume
        volume = volume.transpose(2,0,1)
        r,c = volume[0].shape
        #r = X; c=Y;
        nb_frames = len(volume)
        #print('XYZ',[X,Y,Z])
        #print('rcn',[r,c,nb_frames])
        #prepare density
        maxsize = r*c*nb_frames
        selectfrac=0.01
        selectfrac=1000000./maxsize
        selectfrac = selectfrac/2.
        selectfrac = min(selectfrac,1.)
        selectsize = selectfrac * maxsize
        maskscale = selectsize/nb_frames

        #print('select fraction', selectfrac)
        #print('select size', selectsize)

        myop = min(.8, 20./float(nb_frames))
        myop = 0.8

        vall=[]
        zall=[]
        xall=[]
        yall=[]
        #FRAME-UNIFORM DATA REDUCTION
        for k in range(nb_frames):
            vframe = volume[k].squeeze().T
            vframe=vframe.astype(np.uint8)
            #NOTE XYZ SHOULD BE UINT16 TO ENABLE 65k max imagery
            zframe = k*np.ones((r,c),dtype=np.uint16)
            zframe=zframe.astype(np.uint16)
            xframe,yframe = np.meshgrid(np.arange(r),np.arange(c))
            xframe=xframe.astype(np.uint16)
            yframe=yframe.astype(np.uint16)

            stdfact = 1.
            vmask = vframe>stdfact
            while (np.count_nonzero(vmask) > maskscale) and (stdfact<255.):
                stdfact=stdfact+1.
                vmask = vframe>stdfact
            #print('stdfact ',stdfact)

            #vmask = vflat > 255./2.
            vframe=vframe[vmask]
            zframe=zframe[vmask]
            xframe=xframe[vmask]
            yframe=yframe[vmask]
            #print('xframe size',np.asarray(xframe).shape)
            vall.append(np.copy(vframe).flatten())
            xall.append(np.copy(xframe).flatten())
            yall.append(np.copy(yframe).flatten())
            zall.append(np.copy(zframe).flatten())
            #print('xall size',np.asarray(xall).shape)
        #DISTANCE EVALUATION
        vall = np.concatenate(vall).ravel()
        zall = np.concatenate(zall).ravel()
        xall = vscale*np.concatenate(xall).ravel()
        yall = vscale*np.concatenate(yall).ravel()
        #vall = vall if vall.ndim==1 else vall.flatten()
        #xall = xall if xall.ndim==1 else xall.flatten()
        #yall = yall if yall.ndim==1 else yall.flatten()
        #zall = zall if zall.ndim==1 else zall.flatten()
        oall = np.zeros_like(vall,dtype=float)
        binnum = 10
        #print('len xyz ',xyzarray.shape)
        #print('len xall',xall.shape)
        #print('len yall',yall.shape)
        #print('len zall',zall.shape)
        #print('iterating for ',len(vall))

        dloc = np.asarray([xall[0],yall[0],zall[0]]).astype(float)
        #print('dloc ',dloc)
        dlist = dloc-xyzarray
        #print(dlist)
        dlisttest = np.linalg.norm(dlist,axis=1)
        #print('len dlist ',dlisttest.shape)
        #print(dlisttest)
        #print('MIN: ',np.amin(dlisttest))
        #print('len dlist1 ',dlisttest.shape)
        dlisttest = np.linalg.norm(dlist,axis=0)
        #print('len dlist0 ',dlisttest.shape)
        spreadfactor = max(1., float(min(r,c))/float(nb_frames))
        for i in range(len(vall)):
            i=int(i)
            dloc = np.asarray([xall[i],yall[i],zall[i]]).astype(float)
            dlist = dloc-xyzarray
            dlist[:,-1] = dlist[:,-1]*spreadfactor
            dlist = np.linalg.norm(dlist,axis=1)
            oall[i] = np.amin(dlist)

        sortidx = oall.argsort()
        #sort in ascending order for binning (ignore xall/yall/zall/vall)
        #oall = oall[sortidx]
        bins = np.linspace(np.amin(oall),np.amax(oall),binnum,endpoint=False)
        oallD = np.digitize(oall,bins)
        binnum2 = np.copy(binnum)
        while len(np.unique(oallD))<binnum:
            binnum2 = binnum2+1
            bins = np.linspace(np.amin(oall),np.amax(oall),binnum2,endpoint=False)
            oallD = np.digitize(oall,bins)
        oall = oallD
        #reverse sorting
        #oall = oall[sortidx.argsort()]
        oall = np.exp(-oall) / np.sum(np.exp(np.unique(-oall)))

        #OPACITY-BY-DISTANCE PRINTING
        #print('binnum2: ',binnum2)
        #print('numoptions: ',len(np.unique(oall)))
        #print('options: ',np.unique(oall))
        #print('running...',flush=True)
        pixsize = 2 if selectfrac<0.5 else 1
        for opt in np.unique(oall):
            omask = oall==opt
            #print('newmyop ',opt)
            #print('vlen ',len(vall[omask]))
            #print('running...',flush=True)

            df = pd.DataFrame(np.asarray([xall[omask],yall[omask],zall[omask],vall[omask]]).T, columns=[xname,yname,zname,vname])
            fig.add_trace(go.Scatter3d(x=df[xname],y=df[yname],z=df[zname],mode='markers',showlegend=False,hoverinfo="skip",
                marker=dict(
                size=pixsize,color=df[vname],colorscale='Viridis',opacity=opt,showscale=False
                )))

    fig.update_layout(
        scene=dict(
            xaxis=dict( range=[0,X], backgroundcolor="rgba(50,50,50,.5)", showspikes=False),
            yaxis=dict( range=[0,Y], backgroundcolor="rgba(50,50,50,.5)", showspikes=False),
            zaxis=dict( range=[0,Z], backgroundcolor="rgba(50,50,50,.5)", showspikes=False),
            xaxis_title=xname, yaxis_title=yname, zaxis_title=zname
            ),
        autosize=True #margin=dict(l=0,r=0,b=0,t=0)
        )
    if title:
        fig.update_layout( title_text=title)
    #'''
    fig.write_html('%s/%s.html'%(folder,name), auto_play=False)
    #figBig.write_html('%s/%s.html'%(folder,name))
    print('saved association plot %s'%name)





def print_detections_window(I3,goodlines=[],badlines=[],folder='',savename='temp',args=None,makeimg=1,makevid=0,makegif=1,vs=1,fps=5,amp=0,background=0,
                    ccopt=0, I3loc = [], frames=[-1,-1],
                    imscale=1., median=0, shift=0, datatype='fits',binfactor=1):
    '''    Generate video for a windowd set of images '''
    print('%d 2nd-order meaningful lines, %d other 1st-order meaninful lines'%(len(goodlines),len(badlines)))
    #initialize
    v = prepimg(I3,amp)
    aa,bb,cc,_ = v.shape
    av=int(bb*vs); bv=int(aa*vs); cv=int(cc*vs)
    gifset_all=[]
    gifset_obj=[]

    vol_lines = []
    vol_img = []

    #Get a line thickness for printing
    denom=8
    thick = int(np.ceil(I3.shape[0]/100)/denom)
    if vs<1.:
        thick = int(thick*max(vs,0.5))
    thick = max(thick,2)
    if not os.path.exists(folder):
        os.makedirs(folder)
    #Instantiate videos
    videoname_all='%s/videoAll_%s.avi'%(folder,savename)
    videoname_obj='%s/videoObj_%s.avi'%(folder,savename)
    if makevid==1:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_all = cv2.VideoWriter(videoname_all,fourcc,fps,(av,bv))
        video_obj = cv2.VideoWriter(videoname_obj,fourcc,fps,(av,bv))

    #Instantiate wrirting images
    sub_all = np.copy(v[:,:,-1,:].squeeze())
    sub_all = cv2.resize(sub_all, (av,bv), interpolation = cv2.INTER_CUBIC)
    sub_all = cv2.convertScaleAbs(sub_all)
    sub_img = np.ones_like(sub_all)*int(255/2)
    if background:
        sub_img = np.copy(sub_all)
    viridis_seedB = np.array([range(1,255,int(np.ceil(255/ccopt)))]).astype(np.uint8)
    viridis_seed = np.array([np.linspace(1,255,num=int(ccopt))]).astype(np.uint8)
    viridis_list = cv2.applyColorMap(viridis_seed,cv2.COLORMAP_VIRIDIS)[0]
    print('recording of stars on still image disabled around line 204 (subimg')



    lastrange=np.copy(np.asarray(frames[0]))#np.asarray([0,0])
    fmax = len(frames)
    input_dir=I3loc[0]
    '''
    if median==1:
        I3, _, minimg = import_fits(input_dir,savedata=0,subsample=1,
                    framerange=[-1,-1],scale=imscale,printheader=0,gauss=[1,.6],median=0,
                    shift=shift,binfactor=binfactor, 
                    bigmedian=0,minimg=np.nan)
        #I3 = np.ascontiguousarray(np.array(I3))
        I3 = np.asarray(I3)
        print('confirming large I3 total import shape: ',I3.shape)
        bigmedianimg = np.copy(np.median(I3,axis=0))
        #I3 = I3 - np.expand_dims(bigmedianimg,axis=0)
        #for f in range(I3.shape[0]):
        #    modeval,modecnt = np.unique(I3[f,:,:],return_counts=True)
        #    modeval = modeval[np.argwhere(modecnt==np.max(modecnt))].flatten().tolist()[0]
        #    I3[f,:,:] = I3[f,:,:] - np.copy(modeval)
        #I3 = I3 - np.amin(I3)+1
        #for i in range(I3.shape[2]):
        #    #print('Pre(median/min): ',[np.median(I3[:,:,i]),np.min(I3[:,:,i])])
        #    I3[:,:,i] = I3[:,:,i] - np.median(I3[:,:,i])
        #    I3[:,:,i] = I3[:,:,i] - np.amin(I3[:,:,i]) + 1.
        #    #print('Post(median/min): ',[np.median(I3[:,:,i]),np.min(I3[:,:,i])])
        prep_scale = np.nan#np.mean(I3)+5.*np.std(I3)
        bigmedian=1
    else:
        bigmedianimg=0
        bigmedian=0
        minimg=np.inf
        prep_scale=np.nan


    '''
    I3, _,  = import_fits(input_dir,savedata=0,subsample=1,
                    framerange=[-1,-1],scale=imscale,printheader=0,gauss=[1,.6],median=median,
                    shift=shift,binfactor=binfactor, 
                    bigmedian=0,minimg=np.inf)
    I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))
    prep_scale=np.nan
    print('PRINT_DETECTIONS.PY: WINDOW: VIDEOALL DISABLED')
    for fidx, framerange in enumerate([[0,-1],]):#enumerate(frames):
        #Get frame info
        frameidx=fidx
        offset = max(0,lastrange[1]-framerange[0])
        lastrange = np.copy(framerange)
        if offset>=(framerange[1]-framerange[0]):
            offset=0
        
        '''
        frame1 = lastrange[1] #if frameidx>0 else framerange[0] #min(lastrange[0],framerange[0])
        offset = np.copy(np.abs(framerange[0]-frame1))
        if offset>(framerange[1]-framerange[0]): #if switching windows!
            frame1 = framerange[0]#lastrange = np.copy(framerange)
            offset=0
            break
        offset = offset+1 if offset>0 else offset
        frame2 = framerange[1]
        lastrange = np.copy(np.asarray([frame1,frame2]))
        '''

        #frame1 = lastrange[1] if frameidx>0 else framerange[0] #min(lastrange[0],framerange[0])
        #offset = np.copy(np.abs(framerange[0]-frame1))
        #offset = offset+1 if frameidx>0 else offset
        #frame2 = framerange[1]
        #lastrange = np.copy(np.asarray([frame1,frame2]))


        print('PRINTING FRAMERANGE ',lastrange)
        print('offset ',offset) 
        #Load window for saving 
        input_dir = I3loc[fidx]

        '''
        if datatype=='bin': 
            I3, _ = import_bin(input_dir, savedata=0, subsample=1, 
                framerange=[frame1,frame2], scale=imscale,median=median,shift=shift)
        elif datatype=='fits':
            printheader=0
            if bigmedian==1:
                I3, _ = import_fits(input_dir,savedata=0,subsample=1,
                    framerange=framerange,scale=imscale,printheader=0,gauss=[1,.6],median=median,
                    shift=shift,binfactor=binfactor, 
                    bigmedian=1,bigmedianimg = bigmedianimg,minimg=minimg)
            else:
                I3, _ = import_fits(input_dir, savedata=0, subsample=1, 
                    framerange=framerange, scale=imscale, printheader=0,gauss=[1,.6],median=median,
                    shift=shift,binfactor=binfactor)

        I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))
        #offset = np.abs(lastrange[0]-framerange[0])
        if offset>0:
            I3 = I3[:,:,offset:]
            #headers = headers[offset:]
        '''
        #Prepare image data for printing
        v = prepimg(I3,amp,prep_scale)
        aa,bb,cc,_ = v.shape
        #vs2 = 500./float(max(aa,bb))
        vs2 = vs
        print('vs, vs2: ',[vs,vs2])
        av2=int(bb*vs2); bv2=int(aa*vs2); #cv2=int(cc*vs2)



        #Run video maker
        print('writing : ... %s ... , window %d/%d'%(savename,fidx,fmax))

   
        #for zBASE in range(int(cc)):
        for zBASE in range(int(cc)): 
            z = zBASE #+ lastrange[0] #actual time location over all windows 
            tcolor = float(z)/float(ccopt) #color using all windwos 
            #Create temp data frames
            sub_all = np.copy(v[:,:,zBASE,:].squeeze())
            sub_all = cv2.resize(sub_all, (av,bv), interpolation = cv2.INTER_CUBIC)
            sub_all = cv2.convertScaleAbs(sub_all)
            sub_obj = np.copy(sub_all)
            subwrite_all = np.copy(sub_all)
            subwrite_obj = np.copy(sub_all)
            #create temp vol data frames
            vol_temp = np.copy(sub_all)
            vol_temp = cv2.resize(vol_temp, (av2,bv2), interpolation = cv2.INTER_CUBIC)
            vol_temp = cv2.convertScaleAbs(vol_temp)
            vol_img.append(vol_temp[:,:,0].squeeze())


            #Interpolate (bad)line to draw at z-frame
            #'''
            if len(badlines)>0:
                for k in range(len(badlines)):
                    locline = badlines[k,:6].squeeze()
                    x1,y1,x2,y2,z2 = interp_frame_Z(locline,z)
                    x1*=vs; y1*=vs; x2*=vs; y2*=vs
                    x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                    z2=min(z2,int(ccopt)-1)
                    if all(np.array([x1,y1,x2,y2])==0):
                        continue
                    #sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),(255,0,0),thick)
                    sub_obj = cv2.line(sub_obj,(x1,y1),(x2,y2),(255,0,0),thick)
                    #x1,y1,x2,y2 = interp_frame_draw(locline,z,vs)
                    #if all(np.array([x1,y1,x2,y2])==0):
                    #    continue
                    #sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),(int(255*tcolor),int(255*(1.-tcolor)),0),thick)
                    #sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),viridis_list[int(z)].tolist(),thick)
            #'''
            #Interpolate (good)line to draw at z-frame OVER bad frames
            # so that good frames are not obstructed from visibility
            if len(goodlines)>0:
                for k in range(len(goodlines)):
                    locline = goodlines[k,:6].squeeze()
                    x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                    x1*=vs; y1*=vs; x2*=vs; y2*=vs
                    x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                    
                    if all(np.array([x1,y1,x2,y2])==0):
                        continue
                    #if len(badlines)>0:
                    #sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),(0,0,255),thick)
                    sub_obj = cv2.line(sub_obj,(x1,y1),(x2,y2),(0,0,255),thick)
                    x1,y1,x2,y2 = interp_frame_draw(locline,z,vs,printonly=True)
                    if all(np.array([x1,y1,x2,y2])==0):
                        continue
                    #sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),(int(255*(1-tcolor)),0,int(255*tcolor)),thick)
                    sub_img = cv2.line(sub_img,(x1,y1),(x2,y2),(int(255*(0)),0,int(255)),thick)

            #Draw frames with transparency, to ensure visibility of all lines
            alpha = 0.5
            if alpha<1.:
                #cv2.addWeighted(sub_all,alpha,subwrite_all,1-alpha,0,subwrite_all)
                cv2.addWeighted(sub_obj,alpha,subwrite_obj,1-alpha,0,subwrite_obj)
            else:
                #subwrite_all=sub_all
                subwrite_obj=sub_obj
            #added for possible disabling of vido printing
            if makevid==1:
                #video_all.write(subwrite_all)
                video_obj.write(subwrite_obj)
                #write gif data 
            if makegif==1:
                subwrite_all = cv2.cvtColor(subwrite_all, cv2.COLOR_BGR2RGB)
                subwrite_obj = cv2.cvtColor(subwrite_obj, cv2.COLOR_BGR2RGB)
                #gifset_all.append(subwrite_all)
                gifset_obj.append(subwrite_obj)

    #release video
    print('releasing...')
    if makevid==1:
        video_all.release()
        video_obj.release()
    if makegif==1:
        #save reduced-scale gifs
        pre, ext = os.path.splitext(videoname_all)
        gifname_all = pre+".gif"
        pre, ext = os.path.splitext(videoname_obj)
        gifname_obj = pre+".gif"
        #imageio.mimsave(gifname_all, gifset_all,fps=fps)
        imageio.mimsave(gifname_obj, gifset_obj,fps=fps)
        #imageio.mimsave(gifname_obj, gifset_obj,fps=5)

    #Saving last frame for visualization 
    imageio.imwrite('%s/img_%s.png'%(folder,savename),np.flip(sub_img,-1))
    #imageio.imwrite('%s/img_%s.png'%(folder,savename),np.flip(sub_all[-300:,:300,:],-1))


    #gifset can become an HTML... no 'discrete points' but a volume visualization!
    objlist=[]
    if makevid==1 or makegif==1:
        vol_img = np.dstack(vol_img)
        print('new suggested size: ',[av2,bv2,ccopt])
        print('volimg size: ',vol_img.shape)

        goodlines[:,[0,1,3,4]]*=vs2

        if len(goodlines)>0:
            for k in range(len(goodlines)):
                locline = goodlines[k,:6].squeeze()
                print(locline)
                for z in range(ccopt):
                    z=float(z)
                    x1,y1 = interp_frame_xy(locline,z,double=True,printonly=False)
                    #x1*=vs; y1*=vs; #x2*=vs; y2*=vs
                    #x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                    if not all(np.array([x1,y1])==0):
                      objlist.append([y1,x1,z])
        objlist = np.asarray(objlist)
        ##UPDATE GOODLINES BEFORE
        build_xyzplot(objlist,goodlines,[av2,bv2,ccopt],folder=folder,name='%s_FINALVOL'%savename,volume=vol_img)
    print('saved volume plot %s/%s_FINALVOL.html '%(folder,savename))



    print('Done!\n\n')


def print_detections_window_postastro(I3shape,goodlines=[],badlines=[],folder='',savename='temp',args=None,makeimg=1,makevid=0,makegif=1,vs=1,fps=5,amp=0,background=0,
                    ccopt=0, I3loc = [], frames=[-1,-1],
                    imscale=1., median=0, shift=0, datatype='fits',binfactor=1,rdt=[],title=None,starlines=[]):
    '''    Generate video for a windowed set of images '''
    #print('%d 2nd-order meaningful lines, %d other 1st-order meaninful lines'%(len(goodlines),len(badlines)))
    #initialize
    #print('INPUT DATA:',goodlines.shape)
    #print(goodlines)
    print('PRINTING POINTS PROVIDED IN TDM')
    aa=I3shape[0]; bb=I3shape[1]; cc=I3shape[2]
    av=int(bb*vs); bv=int(aa*vs); cv=int(cc*vs)
    gifset_obj=[]

    vol_lines = []
    vol_img = []

    #Get a line thickness for printing
    denom=8
    thick = int(np.ceil(aa/100)/denom)
    if vs<1.:
        thick = int(thick*max(vs,0.5))
    thick = max(thick,2)
    if not os.path.exists(folder):
        os.makedirs(folder)
    #Instantiate videos
    videoname_obj='%s/videoObj_%s.avi'%(folder,savename)
    if makevid==1:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_obj = cv2.VideoWriter(videoname_obj,fourcc,fps,(av,bv))

    
    lastrange=np.copy(np.asarray(frames[0]))#np.asarray([0,0])
    fmax = len(frames)
    input_dir=I3loc[0]

    I3, _,  = import_fits(input_dir,savedata=0,subsample=1,
                    framerange=[-1,-1],scale=imscale,printheader=0,gauss=[1,.6],median=median,
                    shift=shift,binfactor=binfactor, 
                    bigmedian=0,minimg=np.inf)
    I3 = np.ascontiguousarray(np.array(I3.transpose(1,2,0)))
    #print('PRINTING FRAMERANGE ',lastrange)
    #print('offset ',offset) 
    #Load window for saving 
    #input_dir = I3loc[fidx]

    #Prepare image data for printing
    prep_scale=np.nan
    v = prepimg(I3,amp,prep_scale)
    aa,bb,cc,_ = v.shape
    av=int(bb*vs); bv=int(aa*vs); #cv2=int(cc*vs2)
    av2=av; bv2=bv; vs2=vs;
    #Instantiate wrirting images
    sub_all = np.copy(v[:,:,-1,:].squeeze())
    sub_all = cv2.resize(sub_all, (av,bv), interpolation = cv2.INTER_CUBIC)
    sub_all = cv2.convertScaleAbs(sub_all)

    print('PRINT_DETECTIONS.PY: WINDOW: VIDEOALL DISABLED')
    for fidx, framerange in enumerate([[0,-1],]):#enumerate(frames):
        #Get frame info
        frameidx=fidx
        offset = max(0,lastrange[1]-framerange[0])
        lastrange = np.copy(framerange)
        if offset>=(framerange[1]-framerange[0]):
            offset=0
        

       



        #Run video maker
        print('writing : ... %s ... , window %d/%d'%(savename,fidx,fmax))

   
        #for zBASE in range(int(cc)):
        for zBASE in range(int(cc)): 
            z = zBASE #+ lastrange[0] #actual time location over all windows 
            tcolor = float(z)/float(ccopt) #color using all windwos 
            #Create temp data frames
            sub_all = np.copy(v[:,:,zBASE,:].squeeze())
            sub_all = cv2.resize(sub_all, (av,bv), interpolation = cv2.INTER_CUBIC)
            sub_all = cv2.convertScaleAbs(sub_all)
            sub_obj = np.copy(sub_all)
            subwrite_obj = np.copy(sub_all)
            #create temp vol data frames
            vol_temp = np.copy(sub_all)
            vol_temp = cv2.resize(vol_temp, (av2,bv2), interpolation = cv2.INTER_CUBIC)
            vol_temp = cv2.convertScaleAbs(vol_temp)
            vol_img.append(vol_temp[:,:,0].squeeze())
            '''
            #All stars for calibration from actual a-contrario
            if len(starlines)>0:
                for k in range(len(starlines)):
                    locline =starlines[k,:6].squeeze()
                    x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                    x1*=vs; y1*=vs; x2*=vs; y2*=vs
                    x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                    if all(np.array([x1,y1,x2,y2])==0):
                        continue
                    col=(255,0,255) #stars in magenta
                    sub_obj = cv2.line(sub_obj,(x1,y1),(x2,y2),col,thick)
            '''
            #Actual a-contrario detections
            if len(badlines)>0:
                for k in range(len(badlines)):
                    locline = badlines[k,:6].squeeze()
                    x1,y1,x2,y2 = interp_frame(locline,z,printonly=True)
                    x1*=vs; y1*=vs; x2*=vs; y2*=vs
                    x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                    if all(np.array([x1,y1,x2,y2])==0):
                        #print()
                        continue
                    col=(0,255,0) #trajectories in green
                    sub_obj = cv2.line(sub_obj,(x1,y1),(x2,y2),col,thick*3)



            #Interpolate (good)line to draw at z-frame OVER bad frames
            # so that good frames are not obstructed from visibility
            if len(goodlines)>0:
                for k in range(len(goodlines)):
                    x1=goodlines[k,0]; y1=goodlines[k,1]; z1=goodlines[k,2]
                    if z1<=z:
                        x1*=vs; y1*=vs; 
                        x1=int(x1); y1=int(y1); 
                        if goodlines[k,3]:
                            col=(0,0,255) if z1==z else (128,0,255)
                        else:
                            col=(255,0,0) if z1==z else (255,0,64)
                        sub_obj = cv2.circle(sub_obj,(x1,y1),thick,col,-1)

            #Draw frames with transparency, to ensure visibility of all lines
            alpha = 0.5
            if alpha<1.:
                cv2.addWeighted(sub_obj,alpha,subwrite_obj,1-alpha,0,subwrite_obj)
            else:
                subwrite_obj=sub_obj
            #added for possible disabling of vido printing
            if makevid==1:
                #video_all.write(subwrite_all)
                video_obj.write(subwrite_obj)
                #write gif data 
            if makegif==1:
                subwrite_obj = cv2.cvtColor(subwrite_obj, cv2.COLOR_BGR2RGB)
                gifset_obj.append(subwrite_obj)

    #release video
    print('releasing...')
    if makevid==1:
        video_obj.release()
    if makegif==1:
        #save reduced-scale gifs
        pre, ext = os.path.splitext(videoname_obj)
        gifname_obj = pre+".gif"
        imageio.mimsave(gifname_obj, gifset_obj,fps=fps)
    sub_img = np.asarray(gifset_obj)
    sub_img = np.sum(sub_img,axis=0)
    sub_img[sub_img > 255] = 255
    pre, ext = os.path.splitext(videoname_obj)
    imgname_obj = pre+".png"
    imageio.imwrite(imgname_obj,np.flip(sub_img,-1))      

    #gifset can become an HTML... no 'discrete points' but a volume visualization!
    objlist=[]
    #badlines[:,[0,1,3,4]]*=vs2
    if makevid==1 or makegif==1:
        vol_img = np.dstack(vol_img)
        objlist = np.copy(goodlines).astype(dtype=float)
        rdtidx = np.copy(objlist[:,3]).astype(dtype=bool)
        objlist = objlist[:,:3]
        objlist0 = np.copy(objlist[:,0])
        objlist[:,0] = np.copy(objlist[:,1]) 
        objlist[:,1] = objlist0 
        objlist = objlist[:,:3]
        #objlist[:,[0,1]]*=float(vs2)
        ##UPDATE GOODLINES BEFORE
        build_xyzplot(objlist,badlines,[aa,bb,cc],folder=folder,name='%s_FINALVOL'%savename,volume=vol_img,rdt=rdt,rdtidx=rdtidx,title=title,vscale=1./float(vs2))
    print('saved volume plot %s/%s_FINALVOL.html '%(folder,savename))



    print('Done!\n\n')

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
    savename=''
    folder=''
    goodlines=np.load('%s/goodlines_%s.npy'%(outfolder,savename))
    badlines=np.load('%s/badlines_%s.npy'%(outfolder,savename))
   
    #run detections
    print_detections(np.copy(I3),goodlines,badlines,folder=folder,savename='temp',args=args)

    print('-------ALL DONE-------\n\n')



