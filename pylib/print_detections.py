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
import imageio
import cv2 

def prepimg(img):
    '''    Scale image for printing purposes     '''
    v = np.copy(img)
    #NOTE: img is usually background-subtracted [0,(2^16)-1] uint16
    #Printing is a 3-sigma clipping scaled to [0,255] uint8
    val_max = np.mean(v) + 3*np.std(v)
    v[v>val_max]=val_max
    m = np.min(v)
    v = (v-m)/(val_max-m)
    v = np.clip(v,np.mean(v)-3.*np.std(v),np.mean(v)+3.*np.std(v))
    
    v = np.array((v-v.min())/(v.max()-v.min())*255,np.uint8)
    v = np.stack((v,)*3,axis=-1)
    return np.array(v)

def myinterp(x,x1,x2,y1,y2):
    '''     let x be the z index and 'y' be x or y  '''
    if (x1>x2) or (x<x1) or (x>x2):
        print('error: intep: x1>x2 or x<x1 or x>x2')
        return 
    if (x1==x2) and (y1<y2):
        return y1
    if (x1==x2) and (y1>y2): 
        return y2
    return y1 + (x-x1)*(y2-y1)/(x2-x1)


def interp_frame(locline,z):
    '''    Get x,y line to draw on z-level of video     '''
    x1 = int(locline[1])
    y1 = int(locline[0])
    z1 = int(locline[2])
    x2 = int(locline[4])
    y2 = int(locline[3])
    z2 = int(locline[5])
    if (z<min(z1,z2)):
        return 0,0,0,0
    if z1>z2:
        tempx = np.copy(x1)
        tempy = np.copy(y1)
        tempz = np.copy(z1)
        x1 = np.copy(x2)
        y1 = np.copy(y2)
        z1 = np.copy(z2)
        x2 = tempx
        y2 = tempy
        z2 = tempz
    if (z<z2) and (z1!=z2):
        x2 = int(myinterp(z,z1,z2,x1,x2))          
        y2 = int(myinterp(z,z1,z2,y1,y2))
    return x1,y1,x2,y2

def print_detections(img,goodlines=[],badlines=[],folder='',savename='temp',args=None,makeimg=1,makevid=1,vs=0.25):
    '''    Generate video    '''
    print('%d 2nd-order meaningful lines, %d other 1st-order meaninful lines'%(len(goodlines),len(badlines)))

    #Prepare image data for printing
    v = prepimg(img)
    aa,bb,cc,_ = v.shape
    av=int(aa*vs); bv=int(bb*vs); cv=int(cc*vs)
    gifset_all=[]
    gifset_obj=[]

    #Get a line thickness for printing
    denom=8
    thick = int(np.ceil(v.shape[0]/100)/denom)
    if vs<1.:
        thick = int(thick*max(vs,0.5))
    thick = max(thick,2)

    ## Initialize video writer
    fps = 5.0
    if not os.path.exists(folder):
        os.makedirs(folder)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoname_all='%s/videoAll_%s.avi'%(folder,savename)
    video_all = cv2.VideoWriter(videoname_all,fourcc,fps,(av,bv))
    videoname_obj='%s/videoObj_%s.avi'%(folder,savename)
    video_obj = cv2.VideoWriter(videoname_obj,fourcc,fps,(av,bv))

    #Run video maker
    print('writing : ',savename, '...')

    for z in range(cc):
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
                x1,y1,x2,y2 = interp_frame(locline,z)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),(255,0,0),thick)

        #Interpolate (good)line to draw at z-frame OVER bad frames
        # so that good frames are not obstructed from visibility
        if len(goodlines)>0:
            for k in range(len(goodlines)):
                locline = goodlines[k,:6].squeeze()
                x1,y1,x2,y2 = interp_frame(locline,z)
                x1*=vs; y1*=vs; x2*=vs; y2*=vs
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
                if all(np.array([x1,y1,x2,y2])==0):
                    continue
                sub_all = cv2.line(sub_all,(x1,y1),(x2,y2),(0,0,255),thick)
                sub_obj = cv2.line(sub_obj,(x1,y1),(x2,y2),(0,0,255),thick)

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
        subwrite_all = cv2.cvtColor(subwrite_all, cv2.COLOR_BGR2RGB)
        subwrite_obj = cv2.cvtColor(subwrite_obj, cv2.COLOR_BGR2RGB)
        gifset_all.append(subwrite_all)
        gifset_obj.append(subwrite_obj)


    
    #release video
    print('releasing...')
    video_all.release()
    video_obj.release()
    #save reduced-scale gifs
    pre, ext = os.path.splitext(videoname_all)
    gifname_all = pre+".gif"
    pre, ext = os.path.splitext(videoname_obj)
    gifname_obj = pre+".gif"
    imageio.mimsave(gifname_all, gifset_all,fps=5)
    imageio.mimsave(gifname_obj, gifset_obj,fps=5)

    #Saving last frame for visualization 
    imageio.imwrite('%s/img_%s.png'%(folder,savename),np.flip(sub_all,-1))

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
    goodlines=np.load('%s/goodlines_%s.npy'%(outfolder,savename))
    badlines=np.load('%s/badlines_%s.npy'%(outfolder,savename))
   
    #run detections
    print_detections(np.copy(I3),goodlines,badlines,folder=folder,savename='temp',args=args)

    print('-------ALL DONE-------\n\n')



