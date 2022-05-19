'''
PyCIS - Python Computational Inference from Structure

pylib/run_astrometry.py: Generate right ascension - declination solution.
  Optionally defines error relative to an expected TLE track (under development).

TODO:
  Use CAST-local astrometry solution software in place of Astrometry.net offline pipline 

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
import json
import glob
from datetime import datetime
import time
from time import sleep

import subprocess
from multiprocessing import get_context

import numpy as np
from skimage.feature import match_template 
from astropy.convolution import Gaussian2DKernel
import imageio

from astropy import _erfa as erfa
#import erfa
from astropy.utils import iers
from astropy.utils.iers import conf, LeapSeconds
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, TEME, GCRS, CartesianDifferential,CartesianRepresentation

from sgp4.api import Satrec, SGP4_ERRORS
import sep

from ccsds_ndm.models.ndmxml2.ndmxml_2_0_0_master_2_0 import Tdm as TDMMaster
from ccsds_ndm.ndm_io import NdmIo, NDMFileFormats
from ccsds_ndm.models.ndmxml2 import ndmxml_2_0_0_tdm_2_0 as tdmcore
from ccsds_ndm.models.ndmxml2 import ndmxml_2_0_0_common_2_0 as commoncore
from ccsds_ndm.models.ndmxml2.ndmxml_2_0_0_common_2_0 import AngleType #, AngleUnits
from decimal import Decimal 

from pylib.import_fits import import_fits
from pylib.print_detections import interp_frame_xy, make_kernel, print_detections_window_postastro
from pylib.detect_outliers_agg import remove_matches_block


def fill_data_leg(array,order,robust=False):
    '''Interpolate missing data, for kernel design '''
    n = len(array)
    idx = np.arange(n).astype(float)
    mask = ~np.isnan(array)
    array2 = np.copy(array)[mask]
    idx2 = np.copy(idx)[mask]
    n2a = len(array2)
    if robust:
        diff = np.diff(array2)
        array2 = array2[:-1]
        idx2 = idx2[:-1]
        diff = np.abs(diff)
        diff = (diff - np.mean(diff))/np.std(diff)
        mask = ~(diff > np.quantile(diff,0.75))
        array2 = array2[mask]
        idx2 = idx2[mask]
    n2 = len(array2)
    print('computing over %d/%d (%d)'%(n2,n,n2a))
    idx = 2.*idx/n - 1.
    idx2 = 2.*idx2/n - 1.

    if robust:
        weight = np.ones(n2)
        #empirical fixing by weighting proportional to length
        weight[0] = weight[0]*(float(n2)**2.)
        weight[-1] = weight[-1]*(float(n2)**2.)
        if order>3:
            coeff = np.polynomial.legendre.legfit(idx2,array2,int(order),w=weight)
        else:
            coeff = np.polynomial.polynomial.polyfit(idx2,array2,int(order),w=weight)
    else:
        if order>3:
            coeff = np.polynomial.legendre.legfit(idx2,array2,int(order))
        else:
            coeff = np.polynomial.polynomial.polyfit(idx2,array2,int(order))
    if order>3:
        xfit =  np.polynomial.legendre.legval(idx,coeff)
    else:
        xfit =  np.polynomial.polynomial.polyval(idx,coeff)
    return xfit

def load_astropy():
  ''' load iers/erfa updates for parallelization '''

  #import os, ssl
  #if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
  #  ssl._create_default_https_context = ssl._create_unverified_context
  #Update should be updated once at start
  conf.auto_download = True
  conf.remote_timeout=5
  conf.reload('remote_timeout')
  conf.reload('auto_download')

  print('erfa...')
  table = LeapSeconds.auto_open()
  erfa.leap_seconds.update(table)

  print('iers...')  
  print(iers.IERS_A_URL_MIRROR)
  iers_a = iers.IERS_A.open(iers.IERS_A_URL_MIRROR, cache=True)
  iers.earth_orientation_table.set(iers_a)
  t=Time('2020:001')
  iers_a.ut1_utc(t)
  #print(t.ut1.iso)  
  try:
      iers.IERS.iers_table = iers_a
      LeapSeconds.from_leap_seconds_list(iers.IETF_LEAP_SECOND_URL)
  except:
      print('sleeping to retry ietf leap second url...')
      sleep(60)
      try:
          iers.IERS.iers_table = iers_a
          LeapSeconds.from_leap_seconds_list(iers.IETF_LEAP_SECOND_URL)
      except:
          print('still cannot fetch leap second url, network unreachable')
          quit()

  #Should thereafter always be updated to turn off 
  print('closing...')
  conf.auto_download = False
  conf.remote_timeout=0
  #quit()
  conf.reload('remote_timeout')
  conf.reload('auto_download')

  try: 
    conf.allow_internet=False
    conf.reload('allow_internet')
  except: 
    pass
  return iers_a

def convDMS(x,hours=0,printer=0):
    ''' convert DD angles to DMS (dd:mm:ss.sss) (if hh:mm:ss, use DD=convDD(x)/15.) '''
    sign = x < 0
    x = abs(x)
    mm,ss= divmod(x*3600,60)
    dd,mm = divmod(mm,60)
    sa=int(ss)
    sb = int((ss-sa)*1000)
    if printer==1:
        print('[x,dd,mm,sa,sb]',[x,dd,mm,sa,sb])
    if sign:
      dd = -dd
      return '{0:03}:{1:02}:{2:02}.{3:03}'.format(int(dd),int(mm),sa,sb)
    else:
      return '{0:02}:{1:02}:{2:02}.{3:03}'.format(int(dd),int(mm),sa,sb)

def convDD(x):
    ''' convert DMS (dd:mm:ss.sss) to DD angles  (if hh:mm:ss, use DD=convDD(x)/15.) '''
    X = x.split(":")
    D=int(X[0])
    M=int(X[1])
    S=float(X[2])
    DD = float(D) + float(M)/60. + float(S)/3600.
    return DD 

def make_xyls(ptslist, hdr, name, img=[],imgloc="",frame_zero=-1,
  z=0, shape=[0,0],imscale=1,binfactor=1,importcorr=0,kernel=[],idxlist=[],streaklist=[],starkernel=None,asowid=2):

  ''' 
  Save xyls table of point data.
  As temporary handling of linearity constraint, the local region around a point will be sought for a point matching a kernel,
  gaussian with std=sqrt(asowid) for 'unstreaked' objects, or a star-like kernel oriented with the streak for untracked objects.
  For stars where astrometry failed, a star detection can also be used for a slower but more reliable means of calibration. 

  Input:
    ptslist: [xyz] coordinates of objects of interest at z
    hdr: the header for frame z
    name: the filename to save the table
    img: the image frame, pre-loaded
    imgloc: the location of image frames to load
    frame_zero: OBSOLETE
    z: the frame of interest 
    shape: The x/y dimensions of the image data
    imscale: The scaling factor for data import
    binfactor: The binning factor for data import
    importcorr: OBSOLTETE: a flag used during debug
    kernel: kernel for star detection, provided when wanting source-extracted stars for calibration
    idxlist: a list associating ptslist entries with a given ASO track
    streaklist: a list associating ptslist with a streaked or unstreaked behavior
    starkernel: the star kernel provided for enhancing object detection (namely gaussian object kernel enhancement)
    asowid: the width for a guaussian object detection kernel

  Output:
    [name].xyls: A astropy table with columns 'X', 'Y', 'NFA', 'IDX', 'XCAL', 'YCAL'
      'NFA' is the source brightness or nfa for sorting detections in astrometric calibration
      'XCAL' and 'YCAL' are the a-contrario detection coordinates while 'X','Y' are the source extraction coordinates
    [name]_cal.xyls: Isolates the calibration data above as 'X', 'Y', 'NFA', 'IDX' for separate calibration
  
  '''
  #ensure dimension
  if (ptslist.ndim==1):
    np.expand_dims(ptslist,axis=0)
  lcount = len(ptslist)
  data = Table()
  

  #If no image or imgfiles presented, just save the input data (initial star detection)
  if (len(img)==0) and (not imgloc):
    #Record input data
    data['X'] = ptslist[:,0] 
    data['Y'] = ptslist[:,1]
    data['XCAL'] = data['X']
    data['YCAL'] = data['Y']
    data['NFA'] = ptslist[:,2]
    data['IDX'] = idxlist if len(idxlist)==len(ptslist) else np.zeros((len(ptslist),))
    data['STREAK'] = np.asarray([streaki is not None for streaki in streaklist]).astype(dtype=int)  if len(streaklist)==len(ptslist) else np.zeros((len(ptslist),))
    pass
  else:
    #Solver uses basic source extraction 
    #should resort to this if star aliasing is too severe 
    importlist=[]

    #Import the data files 
    if imgloc:
      #NOTE: ONLY IMPORTCORR=0 IS USED CURRENTLY 
      if importcorr==1 or importcorr==0 or importcorr==-1:
          frame_z = int(frame_zero+z+importcorr)
          img, _ = import_fits(imgloc, savedata=0, subsample=1,
              framerange=[frame_z,frame_z], scale=imscale,binfactor=binfactor)
          sub = img.squeeze()
      else:
          frame_z = int(frame_zero+z)
          img, _ = import_fits(imgloc, savedata=0, subsample=1,
              framerange=[frame_z,frame_z], scale=imscale,binfactor=binfactor)
          sub = img.squeeze()
    else:
        sub = np.copy(img[:,:,int(z)].squeeze())

    CORRECT=[0,0]
    sub_orig = np.copy(sub)
    #If kernel is not provided, this indicates a single object we wish to extract for each pt in ptslist
    #if kernel is provided, we wish to detect all stars 
    if len(kernel)==0:
      objtemp=[]
      #NOTE: MUST PRESEVE SUB
      #Iterate over all object points
      for pt in range(len(ptslist)):

        #Get the local data around the a-contrario detection 
        #Assumption: Use 5% width as local search area (0.25% area ), may reduce later but balances runtime, valid kernel size, and tracking noise
        sub = np.copy(sub_orig)
        xloc = np.mean(np.array(ptslist[pt,0]))
        yloc = np.mean(np.array(ptslist[pt,1]))
        #Streaki is None for tracked objects (or slow moving) and a kernel parameterization for streaked detections
        streaki = streaklist[pt]
        rad = 0.05*min(sub.shape[0],sub.shape[1]) #if (streaki is not None) else 0.01*min(sub.shape[0],sub.shape[1])
        xa = int(max(0,xloc-rad))
        xb = int(min(sub.shape[0],xloc+rad))
        ya = int(max(0,yloc-rad))
        yb = int(min(sub.shape[1],yloc+rad))
        #The source extractor SEP algorithm desires an inverse convention 
        subshapein = np.copy(sub).shape
        sub = sub[ya:yb,xa:xb]
        CORRECT=[ya,xa]
        NDETECT = 10
        thresh=1

        ## Process object detection
        ##GET KERNEL MATCHING DATA
        #match the template using NORMALIZED CROSS-CORRELATION, (PEARSON CORRELATION COEFFICIENT)
        #rho_XY = E[ (X-muX) (Y-muY) ] / (sigX sigY) ; X is the image and Y is the kernel template, integrated over the kernel area over each pixel
        if streaki is not None:
          #If using a 'streaked' or 'untracked/poorly tracked' object with a long feature, consider a star-like kernel oriented with object track
          kernel = make_kernel(streaki[0],streaki[1],streaki[2])#,subtract=True)
          subG = match_template(np.copy(sub),kernel,pad_input=True,mode = 'reflect')
        else:
          #If the object is "unstreaked / well-tracked", use a gaussian kernel and avoid star detections
          kernel = np.asarray(Gaussian2DKernel(np.sqrt(asowid))) 
          subG = match_template(np.copy(sub),kernel,pad_input=True,mode = 'reflect')
          try:
            if starkernel is not None:
              subH = match_template(np.copy(sub),starkernel,pad_input=True,mode = 'reflect')
              subH[subH<0] = 0 #only bother removing positive correlations 
              subG = subG - subH
          except Exception as e:
            print(e)
            print('sub_img shape',sub.shape); print('kernel shape',kernel.shape); 
            print('input shape',subshapein); print('coordinates',[[ya,yb],[xa,xb]])
            quit()

        #Print data if desired 
        if False:
          saveimg = np.copy(subG)
          saveimg = saveimg-np.amin(saveimg)
          saveimg = np.ceil(saveimg*(254./np.amax(saveimg))).astype(np.uint8)
          savek = idxlist[pt] if len(idxlist)==len(ptslist) else 0
          imageio.imwrite('%sCONVOLVEk%d.png'%(name,savek),saveimg)
        #Prepare image for SEP
        subG = np.ascontiguousarray(subG)
        bkg= sep.Background(subG)  
        subG -= bkg
        #select a standard deviation which results in a single detection 
        threshstep = 1.
        #Must adapt due to possible 'streaks' at location 
        while not NDETECT==1:
            while NDETECT>1:
                thresh=thresh+threshstep
                objs = sep.extract(subG,thresh, err = bkg.globalrms)
                NDETECT=len(objs)
            while NDETECT<1:
                thresh=thresh-threshstep
                objs = sep.extract(subG,thresh, err = bkg.globalrms)
                NDETECT=len(objs)
            threshstep/=2.
            if threshstep<1e-3:
                print('WARNING: CANNOT GET ONE SOURCE EXTRACTION ON %d'%frame_z)
                objtemp.append([-1,-1,-1,-1,-1])
                continue
        #Save the detection data for this point
        objtemp.append([objs['x'][0]+CORRECT[1], objs['y'][0]+CORRECT[0], objs['peak'][0],xloc,yloc])
      #Gather all point detections
      objtemp = np.vstack(objtemp) #allows for unexpected sizing 

    else:
      ##STAR DETECTION 
      
      #Get template match 
      insub = np.copy(sub)
      sub = match_template(np.copy(sub),kernel,pad_input=True,mode = 'reflect')
      #Print data if desired 
      if False:
        saveimg = np.copy(sub)
        saveimg = saveimg-np.amin(saveimg)
        saveimg = np.ceil(saveimg*(254./np.amax(saveimg))).astype(np.uint8)
        savek = idxlist[pt] if len(idxlist)==len(ptslist) else 0
        imageio.imwrite('%sCONVOLVEk%d.png'%(name,0),saveimg)
      #Prepare image for SEP
      sub = np.ascontiguousarray(sub)
      bkg= sep.Background(sub) #recompute background and resolve
      sub -= bkg
      #Choose an initial deviation which ensures SEP can run, a very small percentage of the image
      savemean = np.mean(sub)
      savestd = bkg.globalrms
      teststd = 3.0
      sizethresh = 0.001 #5 percent max
      testimg = sub-savemean #no abs, seek only bright
      val = np.quantile(testimg, 1.-sizethresh)
      teststd = val / savestd
      #Run SEP on the image      
      try:
        objs = sep.extract(sub,teststd, err = bkg.globalrms)
      except Exception as e:
        print('z %d cannot solve stars'%z,flush=True)
        print(e)
        datanew = Table()
        datanew['NFA']=[]
        datanew['X']=[]
        datanew['Y']=[]
        datanew['IDX']=[]
        datanew.write('%s.xyls'%name,overwrite=True,format='fits')
        datanew.write('%s_cal.xyls'%name,overwrite=True,format='fits')
        return #abort 
      #Record data 
      objtemp = np.empty((len(objs),5))
      for i in range(len(objs)):
          objtemp[i,0] = objs['x'][i]#+CORRECT[1] 
          objtemp[i,1] = objs['y'][i]#+CORRECT[0]
          objtemp[i,2] = insub[int(objs['x'][i]),int(objs['y'][i])]#objs['peak'][i]
          objtemp[i,3]=objs['x'][i]; objtemp[i,4]=objs['y'][i]

    #SAVE OUTPUT, RECORDING ORIGINAL A-CONTRARIO DETECTION POINTS FOR POST-PROCESSING
    ptstemp=objtemp      
    data['X'] = ptstemp[:,0] 
    data['Y'] = ptstemp[:,1]
    data['NFA'] = ptstemp[:,2] #peak value 
    data['XCAL'] = ptstemp[:,3]
    data['YCAL'] = ptstemp[:,4]
    data['IDX'] = idxlist if len(idxlist)==len(ptstemp) else np.zeros((len(ptstemp),))
    data['STREAK'] = np.asarray([streaki is not None for streaki in streaklist]).astype(dtype=int) if len(streaklist)==len(ptstemp) else np.zeros((len(ptstemp),))

    #filtering of inadmissible points 
    tempfilt = np.copy(np.asarray(data['X'][:]))>0
    datanew = Table()
    datanew['NFA']=data['NFA'][tempfilt]
    datanew['X']=data['X'][tempfilt]
    datanew['Y']=data['Y'][tempfilt]
    datanew['XCAL']=data['XCAL'][tempfilt]
    datanew['YCAL']=data['YCAL'][tempfilt]
    datanew['IDX']=data['IDX'][tempfilt]
    datanew['STREAK']=data['STREAK'][tempfilt]
    del data
    data=datanew
  data.write('%s.xyls'%name,overwrite=True,format='fits')
  #A-CONTRARIO DATA
  datacal = Table()
  datacal['NFA']=data['NFA']
  datacal['X']=data['XCAL']
  datacal['Y']= data['YCAL']
  datacal['IDX'] = data['IDX']
  datacal.write('%s_cal.xyls'%name,overwrite=True,format='fits')
  
def solve_field(name,hdr,scale,binfactor):
  ''' 
  solve name.wcs from name.xyls star list 
  We use the 2MASS and Tycho-2 star catalogs at 30-2000 arcminute diameters.
  Loading additional 2MASS index files enables stronger solving at higher storage cost
  Correct for header information using cropped scale (0-1) and binfactor (>=1).
  '''
  
  w = int(scale[0]) #field width, pixels
  e = int(scale[1]) #field height, pixels
  #Get FOV info
  Lscale = 1.7#5
  Hscale = 1.8#77
  RAscale = 15.
  #print('LHscaleinit: ',[Lscale,Hscale])
  try:
    Lscale = 0.8*hdr['CDELT1']*hdr['NAXIS1'] #deg/pix * pix = FOV
    Hscale = 1.2*hdr['CDELT2']*hdr['NAXIS2']
    RAscale=1.
    #print('LHscalecdelt1: ',[Lscale,Hscale])
  except: 
    RAscale=15.
  try: #FOV*(current length (unbinned)* length) = newFOV
    L = Lscale*(w*binfactor/hdr['XVISSIZE']) #lower bound of image width, degrees
    H = Hscale*(e*binfactor/hdr['YVISSIZE']) #upper bound of image width, degrees
    #print('LHxvissize: ',[L,H])
  except:
    L = Lscale*(w*binfactor/hdr['NAXIS1']) #lower bound of image width, degrees
    H = Hscale*(e*binfactor/hdr['NAXIS2']) #upper bound of image width, degrees
  #L and H provide a range of angular width
  #w/e provide a sense of pixel width
  #we can determine arcseconds per pixel
  #and report to that precision 
  #how much subpixel precision is up for debate, since XY are inexact... 
  #leave this for another day 
    #print('LHnaxis1: ',[L,H])
  #print('LH: ',[L,H])
  if L>=H:
      print('ERROR: SCALE L>=H AT SOLVE-FIELD')
      quit()
  #Correct pointing info and ensure uniform hours/degrees info
  try:
    ra = convDD(hdr['RA'])*RAscale #prior right ascension, degrees
    dec = convDD(hdr['DEC']) #prior right assension, degrees
  except:
    RAscale = 15.     
    if not (":" in str(hdr['CMND_RA'])):
      hdr['CMND_RA'] = convDMS(hdr['CMND_RA']/RAscale)
      hdr['CMND_DEC'] = convDMS(hdr['CMND_DEC'])
    else:
      pass
    try:
      ra = convDD(hdr['CMND_RA'])*RAscale #prior right ascension, degrees
      dec = convDD(hdr['CMND_DEC']) #prior right assension, degrees
    except:
      ra = hdr['CMND_RA']*RAscale #prior right ascension, degrees
      dec =hdr['CMND_DEC'] #prior right assension, degrees
  radius = H*2. #2#search radius, degrees
 
  #run solve, permitting error bounds on solve 
  if os.path.exists('%s.solved'%name):
    os.remove('%s.solved'%name)
  passer=0
  tol=1
  hdulxy = fits.open('%s.xyls'%name)
  dataxy = hdulxy[1].data
  hdulxy.close()
  #xylen = len(dataxy)
  xylen=100 #DONT NEED ALL POINTS, just those with strongest behavior 
  #INCREASE TOLERANCE IN ATTEMPTS TO SOLVE PLATE 
  while passer==0: #no-remove-lines
    #solve_command = "solve-field --no-plots  --no-verify --no-remove-lines --overwrite -E %f  -d 20,50,100 "%(tol)
    #solve_command = "solve-field --no-plots  --no-verify --no-remove-lines --overwrite -E %f  -d 100 --parity neg --odds-to-tune-up 1e5 "%(tol)
    solve_command = "solve-field --no-plots  --no-verify --no-remove-lines --overwrite -E %f  -d %d --parity neg --odds-to-tune-up 1e6 --crpix-center "%(tol,xylen)
    solve_command+= "-X X -Y Y -s NFA -w %d -e %d -u dw -L %f -H %f --ra %f --dec %f --radius %f %s"%(
      w, e, L, H, ra, dec, radius, '%s.xyls'%name)
    #solve_command+= "-X X -Y Y -s NFA -w %d -e %d -u dw -L %f -H %f %s"%(
    #  w, e, L, H, '%s.xyls'%name)
    #solve_command+= "-X X -Y Y -s NFA -w %d -e %d -u dw -L %f -H %f %s"%(
    #  w, e, L, H, '%s.xyls'%name)
    solve_result = subprocess.run(solve_command,shell=True,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL )
    if solve_result.returncode:
        print(solve_result.returncode)
        print('ERROR: SOLVE-FIELD NOT FOUND, TRY RUNNING SETUP.SH')
        #quit()
    passer=1
    if os.path.exists('%s.solved'%name):
      passer=1
    else:
        if tol==1:
            tol+=1
        else:
            tol*=tol
        if tol>min(w*.01,10): #at once percent
            passer=1

def convert(starname, objname, hdrin, frame,usetle):
  ''' 
   convert object data from pixel to radec coords using stellar astrometry 
   HARDCODED  LOCATION AND TIMEZONE 
  '''
  convert_command = 'wcs-xy2rd -w %s -i %s -o %s'%('%s.wcs'%starname, '%s.xyls'%objname, '%s.rdls'%objname)
  convert_result = subprocess.run(convert_command, shell=True) #convert rso data
  convertcal_command = 'wcs-xy2rd -w %s -i %s -o %s'%('%s.wcs'%starname, '%s_cal.xyls'%objname, '%s_cal.rdls'%objname)
  convertcal_result = subprocess.run(convertcal_command, shell=True) #convert rso data
  #update header
  hdr = hdrin.copy()
  hdulrd = fits.open('%s.rdls'%objname)
  datard = hdulrd[1].data
  hdulrd.close()
  hdulxy = fits.open('%s.xyls'%objname)
  dataxy = hdulxy[1].data
  hdulxy.close()
  #Manual label conversion
  hdulrdcal = fits.open('%s_cal.rdls'%objname)
  datardcal = hdulrdcal[1].data
  hdulrdcal.close()
  racal =  datardcal[0].field(0)
  deccal=  datardcal[0].field(1)

  numrows = 0
  #Load ICRS coordinates, telescope location, and time with exposure offset
  try:   
    obstime=Time(hdr['DATE-OBS'],format='isot',scale='utc')+TimeDelta(hdr['EXPOSURE']/2./86400.)
    loclat = hdr['TELLAT']
    loclon = hdr['TELLONG']
    localt = hdr['TELALT']
    location = EarthLocation(lat=loclat*u.degree,
        lon=loclon*u.degree,height=localt*u.meter)
  except:
    dayA = hdr['DATE_OBS'].split(" ")[0]
    dayA = dayA.split("/")    
    dayB = hdr['TIME-OBS'].split(" ")[0]
    dayB = dayB.split(":")
    day = datetime(year=2000+int(float(dayA[2])),month=int(float(dayA[1])),day=int(float(dayA[0])),
            hour=int(float(dayB[0])),minute=int(float(dayB[1])),second=int(float(dayB[2])) )
    tdelta =hdr['EXPTIME']#.split(" ")[0]
    loclat = 38.8871277
    loclon = -104.2015867
    localt = 1832 #m
    #timezonehrs = -6.# / 24.
    location = EarthLocation(lat=loclat*u.degree,
              lon=loclon*u.degree,height=localt*u.meter)
    obstime=Time(day,format='datetime',scale='utc')+TimeDelta(tdelta/2./86400.)
    #location.representation_type='spherical'

  #ITERATE POINTS
  for i in range(len(datard)):
    xloc = dataxy[i]['X']
    yloc = dataxy[i]['Y']
    peakloc = dataxy[i]['NFA']
    iloc = dataxy[i]['IDX']
    sloc = dataxy[i]['STREAK']
    xcal = dataxy[i]['XCAL']
    ycal = dataxy[i]['YCAL']


    ra =  datard[i].field(0)
    dec=  datard[i].field(1)
    '''
    per http://data.astrometry.net/4200 :
    solutions are by default in  the J2000.0 ICRS reference system.
    We convert celestial solutions to FK5 for header agreement
    We convert apparent  solutions to TEME for prior agreement
    '''
    c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs',
          obstime=obstime,location=location)
    c_fk5 = c_icrs#.transform_to('fk5')
    ra_fk5 = c_fk5.ra.value
    dec_fk5 = c_fk5.dec.value
    ra_fk5 = ra
    dec_fk5 = dec
    c_teme = c_icrs.transform_to('teme')
    c_teme.representation_type='spherical'
    ra_teme = c_teme.lon.value
    dec_teme = c_teme.lat.value
  
    #IF USING TLE, CONVERT FROM J2000 PROJECTION TO GCRS LOS CORRECTION 
    usetle==0
    if usetle==1:
        print('applying gcrs transform')
        c_gcrs = c_icrs.transform_to(GCRS(obstime=obstime))
        l_gcrs = location.get_gcrs(obstime=obstime)
        d_x = (c_gcrs.cartesian.x * 38095*u.km + l_gcrs.cartesian.x.to(u.km))
        d_y = (c_gcrs.cartesian.y * 38095*u.km + l_gcrs.cartesian.y.to(u.km))
        d_z = (c_gcrs.cartesian.z * 38095*u.km + l_gcrs.cartesian.z.to(u.km))
        test_vec = CartesianRepresentation([d_x,d_y,d_z])
        test_gcrs=GCRS(test_vec, obstime=obstime)
        try:
            ra_fk5 = test_gcrs.ra.value
            dec_fk5 = test_gcrs.dec.value
        except Exception as e: 
            print('GCRS RA FAILURE',e)

    obstime.format='fits'
    pytime=obstime.value
    try:
      Lscale = hdr['CDELT1']
      Hscale = hdr['CDELT2']
      RAscale=15.
    except: 
      RAscale=15.
    #SAVE J2000 MANUAL LABEL OR GCRS AGAINST TLE
    if usetle==1:
        hdr.set('RA_PY%d'%i, convDMS(ra_fk5/RAscale), 'PyCIS target RA FK5 [hour]')
        hdr.set('DEC_PY%d'%i, convDMS(dec_fk5), 'PyCIS target Dec FK5 [deg]')
    else:
        hdr.set('RA_PY%d'%i, convDMS(ra_fk5/RAscale), 'PyCIS target RA GCRS [hour]')
        hdr.set('DEC_PY%d'%i, convDMS(dec_fk5), 'PyCIS target Dec GCRS [deg]')
    hdr.set('AR_PY%d'%i, convDMS(ra_teme/RAscale), 'PyCIS target RA apparent TEME [hour]')
    hdr.set('AD_PY%d'%i, convDMS(dec_teme), 'PyCIS target Dec apparent TEME [deg]' )
    hdr.set('X_PY%d'%i, xloc, 'PyCIS target X-position [pixels of original frame]')
    hdr.set('Y_PY%d'%i, yloc, 'PyCIS target Y-position [pixels of original frame]')
    hdr.set('X0_PY%d'%i, xcal, 'PyCIS cal target X-position [pixels of original frame]')
    hdr.set('Y0_PY%d'%i, ycal, 'PyCIS cal target Y-position [pixels of original frame]')
    hdr.set('IDX_PY%d'%i, iloc, 'PyCIS target associated track index')
    hdr.set('STR_PY%d'%i, sloc, 'PyCIS target associated track boolean streak-like flag')

    hdr.set('NFA_PY%d'%i, dataxy[i].field(2), 'PyCIS centerline NFA')
    numrows+=1

  #Store additional info to header
  hdr.set('NUM_PY', numrows, 'PyCIS number of detections')
  hdr.set('TIME_PY', pytime, 'PyCIS: utc obs time given integration time')
  hdr.set('LAT_PY', loclat, 'PyCIS: site latitude (deg)')
  hdr.set('LON_PY', loclon, 'PyCIS: site longitude (deg)')
  hdr.set('ALT_PY', localt, 'PyCIS: site altitude (m)')
  #Store manual claibration info (J2000 only)
  cal_icrs = SkyCoord(ra=racal*u.degree, dec=deccal*u.degree, frame='icrs',
        obstime=obstime,location=location)
  cal_fk5 = cal_icrs.transform_to('fk5')
  racal_fk5 = cal_fk5.ra.value
  deccal_fk5 = cal_fk5.dec.value
  hdr.set('R0_PY', convDMS(racal_fk5/RAscale), 'PyCIS calibration RA FK5 [hour]')
  hdr.set('D0_PY', convDMS(deccal_fk5), 'PyCIS calibration Dec  FK5 [deg]' )
  return hdr

def convertXYONLY(starname, objname, hdrin, frame,usetle):
  ''' 
   convert object data from pixel to radec coords using stellar astrometry 
   HARDCODED  LOCATION AND TIMEZONE 
  '''
  #update header
  hdulxy = fits.open('%s.xyls'%objname)
  dataxy = hdulxy[1].data
  hdulxy.close()
  numrows = 0
  #Load ICRS coordinates, telescope location, and time with exposure offset
  #dataxy['FRAME'] = frame*np.ones_line(dataxy['X'])
  #for i in range(len(dataxy)):
  #  hdr.set('NFA_PY%d'%i, dataxy[i].field(2), 'PyCIS centerline NFA')
  #  numrows+=1
  hdr = dataxy
  return hdr

def null_hdr(hdrin,usetle):
  ''' update header for missed detections '''
  if usetle==-1:
    hdr = Table()
    hdr['X'] = []
    hdr['Y'] = []
    hdr['NFA'] = []
    return hdr
  try:
    hdr = hdrin.copy()
    hdr.set('NUM_PY', 0, 'PyCIS number of detections')
  except: 
    hdr = Table()
    hdr['X'] = []
    hdr['Y'] = []
    hdr['NFA'] = []
  return hdr

def cleanup(folder):
  ''' remove temp files'''
  imlistwcs = glob.glob('%s/*.wcs'%(folder))
  imlistxy = glob.glob('%s/*.xyls'%(folder))
  imlistaxy = glob.glob('%s/*.axy'%(folder))
  imlistcorr = glob.glob('%s/*.corr'%(folder))
  imlistmatch = glob.glob('%s/*.match'%(folder))
  imlistrdls = glob.glob('%s/*.rdls'%(folder))
  imlistsolved = glob.glob('%s/*.solved'%(folder))
  for im in imlistwcs:
    os.remove(im) 
  for im in imlistxy:
    os.remove(im) 
  for im in imlistaxy:
    os.remove(im) 
  for im in imlistcorr:
    os.remove(im) 
  for im in imlistmatch:
    os.remove(im) 
  for im in imlistrdls:
    os.remove(im) 
  for im in imlistsolved:
    os.remove(im) 

def run_solve(starname,objname,hdr,scale,z,iers_a,binfactor,usetle):
    #solve plate
    if not usetle==-1:
        iers.earth_orientation_table.set(iers_a)
        solve_field(starname, hdr, scale,binfactor)
        if not os.path.exists('%s.solved'%starname):
            #print('starsolve z %d failed, continuting'%(z),flush=True)
            return null_hdr(hdr,usetle), z, False
        else:
            pass#print('starsolve z %d passed, continuting'%(z),flush=True)
        #solve stars and update header
        #print('pass successful')
        hdrnew = convert(starname, objname, hdr, z, usetle)
    else:
        print('z %d returning XY'%(z),flush=True)
        hdrnew = convertXYONLY(starname, objname, hdr, z, usetle)
    return hdrnew,z, True


def build_kernels(hlen,goodlines,badlines,input_dirs):
  '''Interpolate kernel parameter data if there are missing or poorly-solved elements'''
  maskset = np.nan*np.ones((hlen,3))
  if 1==1:
      for z in range(hlen):
        if (len(badlines)==0):# or (len(goodlines)==0):
          continue

        starlist = []
        tempstarlines=[]
        for k in range(len(badlines)):
          locline2 = badlines[k,:].squeeze()
          locline = badlines[k,:6].squeeze()
          x,y= interp_frame_xy(locline,z,double=True,star=0)
          if not all(np.array([x,y])==0):
            tempstarlines.append(locline2)
            val = badlines[k,-1]
            starlist.append([x,y,badlines[k,-1],0])
        if len(starlist)==0:
            maskset[int(z),:] = [np.nan,np.nan,np.nan]
            continue

        if not input_dirs:
            pass
        else:
            if len(starlist)==1:
                lines = np.asarray(tempstarlines).reshape(1,-1)
            else:
                lines = np.vstack(tempstarlines)
            k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
            el = np.arccos((lines[:,5]-lines[:,2])/k_len)
            az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))

            elmask = (el*180./np.pi) > (90.-22.5)
            if np.count_nonzero(elmask)>1:
                medlen = np.median(k_len[elmask])
                kmask = k_len>medlen
                elmask = np.logical_and(elmask,kmask)
                if np.count_nonzero(elmask)>1:
                    #az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))
                    medaz = np.median(az[elmask])
                    medwid = np.median(lines[elmask,6])
                    #print('kernel length, width, az: ',[medlen,medwid,medaz*180./np.pi])
                    maskset[int(z),:] = [medlen,medwid,medaz]
  ##INTERPOLATE MASKSET COLUMNS
  maskset[~np.isnan(maskset[:,2]),2] = np.unwrap(maskset[~np.isnan(maskset[:,2]),2],period=np.pi/2)
  print('interpolating...')
  num_pass = np.count_nonzero(~np.isnan(maskset[:,0]))
  num_pass = int(np.ceil(num_pass/10)) #10th-order overdeterminism
  num_order = min(num_pass, 5) #always use less than 5 order
  print('SOLVING WITH ORDER 5 OUT OF MAX ORDER ',num_pass)
  maskset[:,0] = fill_data_leg(maskset[:,0].squeeze(),num_order,True)
  maskset[:,1] = fill_data_leg(maskset[:,1].squeeze(),num_order,True)
  maskset[:,2] = fill_data_leg(maskset[:,2].squeeze(),num_order,True)
  maskset[maskset[:,0]<5.,0] = 5.
  maskset[maskset[:,1]<2.,1] = 2.
  return maskset


def convert_json_tdm(data, hdrs, write_path,tol):
  '''
  Build CCSDS-format TDMs for each associated Ra/Dec track.
  Reject those points which are not within 'tol' of the a-contrario solution,
  and if more than half the points are missed reject as a false-positive a-contrario detection.
  TODO: This can exist after improved non-linear a-contrario association, but should not be necessary. 

  Input: 
    data: json-format radec detection from main function 
    hdrs: headers containing important data
    write_path: where to save the xls-format TDM
    tol: the image-frame L2 distance a source extraction point can be from the a-contrario detection when recording 

  Output: 
    [write_path]_obj[X].xls: The ccsds TDM file for associated track X
    printdata: [x,y,z,rdt]: the source extraction data for printing, with a 'rdt' boolean flag whether this element is included in rdtdata
    rdtdata: [right ascension dd:mm:ss.sss, declination dd:mm:ss.sss, time utc] for each printdata element with 'rdt'=True
    title: A string for printing purposes, including the ASO NORAD ID and hdrs[0] initial Timestamp
  '''
  #Instantiate
  hdr = hdrs[0]
  tdm = TDMMaster()
  #Main metadata
  myheader = tdmcore.TdmHeader()
  myheader.comment = ["PyCIS TDM Product version 2022-02-23"],
  myheader.creation_date = datetime.utcnow().isoformat()
  myheader.originator="UT-ASTRIA"
  tdm.header = myheader
  #Get title information of object/time for printing purposes 
  title_obj = str(hdr["OBJECTID"])
  try:   
    obstime=str(Time(hdr['DATE-OBS'],format='isot',scale='utc')+TimeDelta(hdr['EXPOSURE']/2./86400.))
  except:
    dayA = hdr['DATE_OBS'].split(" ")[0]
    dayA = dayA.split("/")    
    dayB = hdr['TIME-OBS'].split(" ")[0]
    dayB = dayB.split(":")
    day = datetime(year=2000+int(float(dayA[2])),month=int(float(dayA[1])),day=int(float(dayA[0])),
            hour=int(float(dayB[0])),minute=int(float(dayB[1])),second=int(float(dayB[2])) )
    tdelta =hdr['EXPTIME']#.split(" ")[0]
    obstime=str(Time(day,format='datetime',scale='utc')+TimeDelta(tdelta/2./86400.))
  title_time=obstime
  title = "Detections for ASO %s on %s."%(title_obj,title_time)  
  title="%s(Measurements in red, a-contrario trajectories in green, purple/yellow colormap of pixel intensity.)"%title

  #Get number of segments 
  idxmax=0
  for row in data:
    #print(row)
    if row["num_objects"]>0:
      idx = int(row["index"])
      idxmax = idx if idx>idxmax else idxmax


  printdata=[]
  rdtdata=[]
  #FOR EACH ASSOCIATED TRACK:
  for iloc in range(1,1+idxmax):
      mybody = tdmcore.TdmBody()
      mysegmentlist = []

      #Build metadata
      mysegment = tdmcore.TdmSegment()
      mymeta = tdmcore.TdmMetadata()
      mymeta.time_system = commoncore.TimeSystemType('UTC')
      mymeta.participant_1 = '%s_OBS%d'%(str(hdr["OBJECTID"]),iloc)
      mymeta.participant_2 = "TRCK1 (Lat: %f, Lon: %f, Alt: %f km)"%(hdr["TELLAT"],hdr["TELLONG"],hdr["TELALT"]/1000.)
      mymeta.mode = tdmcore.ModeType("SEQUENTIAL")
      mymeta.path = "1,2"
      mymeta.angle_type = tdmcore.AngleTypeType('RADEC')
      mymeta.reference_frame = tdmcore.RefFrameType("EME2000")
      mysegment.metadata = mymeta
      mysegmentlist.append(mysegment)
      mybody.segment = mysegmentlist
      
      framelist=[]
      myobslist = []
      err=0
      subdata=[] #XYZ information 
      subrdt=[]  #Right ascension, declination, time information 

      #Build body data 
      for row in data:
        idx = int(row["index"])
        if idx==iloc and row["num_objects"]>0:

          #Check if source extraction detection is within bounds from a-contrario result 
          angles = row["object_coords"]
          framelist.append(row["frame"])
          sloc=row["streaklike"] 
          erri = ((row['x']-row['xcal'])**2.+(row['y']-row['ycal'])**2.)**.5
          err+=erri
          #print('idx %d frame %d: err: %.2f / %.2f... xy=[%.2f,%.2f], streak %d'%(idx,row["frame"],erri,tol,row['x'],row['y'],sloc))
          if (sloc==0) and (erri>tol): #this filtering should really only be necessary for the gaussian kernel matching
            subdata.append([row['x'],row['y'],row['frame'],False])
            continue
          subdata.append([row['x'],row['y'],row['frame'],True])
          #store right ascension as one row, decimal degrees
          ra = float(angles[0])*180./np.pi
          myobs = tdmcore.TrackingDataObservationType()
          myobs.angle_1 = AngleType(Decimal(ra))#, AngleUnits.DEG )
          myobs.epoch = str(row["time"])
          myobslist.append(myobs)
          #store declination as a second row, decimal degrees
          dec = float(angles[1])*180./np.pi
          myobs = tdmcore.TrackingDataObservationType()
          myobs.angle_2 = AngleType(Decimal(dec))#, AngleUnits.DEG )
          myobs.epoch = str(row["time"])
          myobslist.append(myobs)
          subrdt.append([str(convDMS(ra)),str(convDMS(dec)),str(row["time"])])

        #Save body       
        mydata = tdmcore.TdmData()
        mydata.observation = myobslist
        #mybody.segment[iloc-1].data = mydata 
        mybody.segment[0].data = mydata 

      #Check that at least half the points have passed tolerance in order to save measurements as valid
      density = float(np.count_nonzero(np.vstack(subdata)[:,3]))/float(len(framelist))
      err = np.mean(err)#**.5
      tdm.body = mybody
      print('OBJECT %d HAS LENGTH %d (%d-%d), DENSITY %.4f, err %.4f'%(iloc,max(framelist)-min(framelist),min(framelist),max(framelist),density,err))
      density_tol = .5 #how dense should the track be?  If not dense enough, should just reject
      if density>density_tol:
        printdata.extend(subdata)
        rdtdata.extend(subrdt)
        #Each associated track gets its own CCSDS to interface better with ASTRIAGraph 
        seg_path = os.path.splitext(write_path)[0]+'_obj%d.xml'%iloc
        NdmIo().to_file(tdm, NDMFileFormats.XML, seg_path)
      else:
        #Data which is not written to TDM should still be recorded for printing purposes
        for s in subdata:
          s[3] = False
        printdata.extend(subdata)
  return np.vstack(printdata),rdtdata,title

def run_astrometry_sub(zlist, infolder, savename,goodlines,badlines,headers,headersnew,usetle,imgshape,input_dirs,maskset,extrap,
      jsonname,skipifexist,useimg,imgastro,img,frame_zero,imscale,binfactor,importcorr,cropscale,iers_a):
  ''' 
  A helper function for running object detection and astrometry on a given set of 'zlist' frames, 
  for easily switching between using a-contrario results and cross-correlation results for star calibration 
  ''' 
  folder = '%s_work'%infolder
  iterable=[]
  ## FOR EACH FRAME OF INTEREST
  for z in zlist:
    #Specify the files we will use for storing data
    kernel=None
    starname='%s/%s_star%d'%(folder,savename,z)
    objname='%s/%s_obj%d'%(folder,savename,z)
    hdr = headers[z]
    #Make sure there is sufficient star/object data to consider
    if (len(badlines)==0) or (len(goodlines)==0):
      print('ERROR: lines empty, check results: badlines %d goodlines %d'%(len(badlines),len(goodlines)))
      headersnew[z] =null_hdr(hdr,usetle)
      continue

    #create star list
    starlist = []
    tempstarlines=[]
    for k in range(len(badlines)):
      locline2 = badlines[k,:].squeeze()
      locline = badlines[k,:6].squeeze()
      #x,y= interp_frame_xy(locline,z,star=3)
      x,y= interp_frame_xy(locline,z,double=True,star=0)
      if not locline[2]==locline[5]:
        if (z==locline[2]) or (z==locline[5]):
          continue 
      if ((x<=0) | (x>=imgshape[0])) | ((y<=0) | (y>=imgshape[1])):
        x=0; y=0;
      if not all(np.array([x,y])==0):
        tempstarlines.append(locline2)
        starlist.append([x,y,badlines[k,-1],0])

    #Abort if there are not enough valid stars 
    if (len(starlist)==0):
      headersnew[z] =null_hdr(hdr,usetle)
      print('EMPTY STARLIST frame %d'%z)
      continue
    #Build the star kernel for this frame
    if not imgastro==0:
        medlen = maskset[int(z),0]
        medwid = maskset[int(z),1]
        medaz = maskset[int(z),2]
        if (not np.isnan(medlen)):
            kernel = make_kernel(medlen,medwid,medaz)

    #create obj list
    objlist = []
    idxlist=[]
    streaklist=[]
    #Find all a-contrario detections intersecting this frame, with a flag for extrapolating all lines 
    for k in range(len(goodlines)):
      locline = goodlines[k,:6].squeeze()
      x,y = interp_frame_xy(locline,z,extrap=extrap,shape=imgshape,double=True,printdetail=True)
      if ((x<0) | (x>imgshape[0])) | ((y<0) | (y>imgshape[1])):
        x=0; y=0;
     
      if not all(np.array([x,y])==0):
        #Record 1) corrdinates, 2) association index, 3) streak kernel parameters if the object is not likely gaussian PSF
        objlist.append([x,y,goodlines[k,-1]])
        idxlist.append(k+1)
        xylen = ((goodlines[k,0]-goodlines[k,3])**2. + (goodlines[k,1]-goodlines[k,4])**2.)**.5
        if xylen>(.2*min(imgshape[0],imgshape[1])):
            slen=  np.mean(maskset[:,0]) #0.01*min(imgshape[0],imgshape[1])
            swid = np.mean(maskset[:,1]) #goodlines[k,6]/2.
            saz = np.arctan2((locline[3]-locline[0]) , (locline[4]-locline[1]))
            streaklist.append([slen,swid,saz])
        else:
            streaklist.append(None)
      
    #account for no-detection case
    if (len(starlist)==0) or (len(objlist)==0):
      #print('EMPTY OBJECT, FRAME %d'%z)
      headersnew[z] =null_hdr(hdr,usetle)
      continue

    #IMGASTROSETTINGS
    #0 - all pycis
    #1 - source extract stars, pycis asos
    #2 - all source extract
    #3 - pycis stars, source extract asos
    if not(os.path.exists(jsonname) and skipifexist):
        if usetle==-1 or (useimg==0 and (imgastro==0 or imgastro==3)):
          make_xyls(np.array(starlist),hdr, starname, z=z, shape=imgshape)#, img=img,z=z)
        else:
          if not input_dirs:
              make_xyls(np.array(starlist),hdr, starname, img=img,z=z, shape=imgshape)
          else:
              if 1==0:#imgastro==2:
                  make_xyls(np.array(starlist),hdr, starname,z=z, shape=imgshape)
              else:
                  make_xyls(np.array(starlist),hdr, starname, imgloc=input_dirs[0],frame_zero=frame_zero,z=z, shape=imgshape,imscale=imscale,binfactor=binfactor,importcorr=importcorr,kernel=kernel)
    if not(os.path.exists(jsonname) and skipifexist):
        if not (imgastro==2 or imgastro==3):
            make_xyls(np.array(objlist), hdr, objname, z=z, shape=imgshape,idxlist=idxlist,starkernel=kernel,asowid=np.mean(maskset[:,1]))
        else:
            make_xyls(np.array(objlist), hdr, objname, z=z, shape=imgshape,idxlist=idxlist,imgloc=input_dirs[0],frame_zero=frame_zero,imscale=imscale,binfactor=binfactor,importcorr=importcorr,kernel=[],streaklist=streaklist,starkernel=kernel,asowid=np.mean(maskset[:,1]))
        iterable.append((starname,objname,hdr,cropscale,z,iers_a,binfactor,usetle))
  return iterable,headersnew

def run_astrometry(img,goodlines, badlines, headers, scale=1., folder='temp',tdmfolder='temp',savename='temp',
  makejson=0,tle=[],binfactor=1,vs=0.25,imgastro=0,subprocess_count=10,
  I3loc=[],frames=[],
  imscale=1.,median=1,shift=0,datatype='fits',kernel=[],imgshape=[0,0,0]):
  '''
  Launch Astrometry.net plate solving, group and convert a-contrario detections, 
  and update headers for updating fits files or JSON output
  The results of interest are a CCSDS-format Tracking Data Message for each associated track detected, 
  with a plotly HTML plot including the measured calibrated data and a-contrario trajectory detections.

  Input: 
    img -        image data for optional sorting
    goodlines -  line features corresponding to 2nd-order-meaninful detected objects 
    badlines -   line features corresponding to 1st-order-meaninful stars and noise
    headers -    original headers from fits files
    scale -      portion of cropped image (1 uncropped)
    folder -     where to save temporary files
    savename -   base name for temporary files
    makejson -   (0/1) save file of astrometric results for positioning and PR analysis 
    tle -        tle to use for PR analysis ([] to use HARDCODED manual label)
    binfactor -  binning factor of image to correct data/header discrepancies
    imgastro -   (0/1) use img to sort stars by intensity rather than centerline-NFA

  Output: 
    headersnew - updated headers with detection ra/dec and nfa information
    record_name.json - ra/dec track, and error data for PR analysis
    *obj[X].xls - CCSDS-Format tracking data message for associated track X
    *ASTRO*html - a plotLY HTML interactive plot of the measured calibrated data and a-contrario trajectory detections.

  '''
  
  ## BY DEFAULT, WE WILL ATTEMPT SOURCE EXTRACTION ON THE EXTRAPOLATED A-CONTRARIO DETECTIONS
  #  USING A FILTER IN CCSDS TDM CONSTRUCTION TO REMOVE NOISY A-CONTRARIO LINES AND INVALID SOURCE EXTRACTION POINTS
  extrap=True
  print('IMGSHAPE',imgshape)
  print(goodlines)
  print('Filtering lines without azimuth filter.  %d ->'%(len(goodlines)), end=' ')
  #SOURCE EXTRACTION WILL USE A 5% WIDTH DOMAIN, TDM MEASUREMENT FILTER WILL USE A 1% WIDTH DOMAIN 
  septol=0.05*min(imgshape[0],imgshape[1])
  errtol=0.01*min(imgshape[0],imgshape[1])
  disttol = 0.001*min(imgshape[0],imgshape[1])
  #ASSOCIATE ALL LINES WHICH INTERSECT WHEN EXTRAPOLATED OVER SEPTOL
  #handles implementation error in current linearized association 
  goodlines,_ = remove_matches_block(goodlines, goodlines,len(headers),identical=True,septol=septol,azfilter=False,disttol=disttol)
  print('%d'%(len(goodlines)))

  #Handling of window-based analysis, TLE flags, astropy data
  cropscale=np.copy(scale)
  for importcorr in [0,]:
      print('\n\nIMPORTCORR ',importcorr)
      input_dirs=[]
      frame_zero = -101
      if I3loc and frames:
        lastrange=np.asarray([-101,-101])
        fmax = len(frames)
        for fidx, framerange in enumerate(frames):
            #Get frame info
            frame1 = max(lastrange[0],framerange[0])
            if frame_zero==-101:
                frame_zero=np.copy(frame1)
            frame2 = framerange[1]
            lastrange = np.asarray([frame1,frame2])
            input_dirs.append(I3loc[fidx])
      #Interpolate each frame
      headersnew = headers.copy()
      imgold = np.copy(img)
      iterable=[]
      useimg=0
      if not tle:
          usetle=0
      elif not isinstance(tle,list):
          usetle=-1
      else:
          usetle=1
      if not usetle==-1:
          iers_a=load_astropy()
      else:
          iers_a = 0
      print('USE TLE: ',usetle)
      print('USE IMG (HARDCODE): ',useimg)


  #IF JSON EXISTS
  skipifexist=0
  jsonname='%s/record_%s.json'%(folder,savename)

  #interpolate kernel data
  if not imgastro==0:
      maskset = build_kernels(len(headers),goodlines,badlines,input_dirs)

  printdata=[]
  #TODO: OBSOLETE: This while loop will run exactly once currently. REmove
  while useimg<=1:

    #if True: #not(os.path.exists(jsonname) and skipifexist):

    #Build an iterable for astrometry using the raw a-contrario star detections 
    subtime=time.time()
    iterable,headersnew = run_astrometry_sub(range(len(headers)), folder, savename,goodlines,badlines,headers,headersnew,usetle,imgshape,input_dirs,maskset,extrap,
        jsonname,skipifexist,useimg,imgastro,img,frame_zero,imscale,binfactor,importcorr,cropscale,iers_a)
    print('FIRST RUN: ITERABLE IS LENGTH: ',len(iterable))
    subtime-=time.time()
    subtime/=60.
    print('Runtime: %.2f'%subtime)
    subtime=time.time()
    len_all_detections = len(iterable)
    #Run the astrometric plate solver 
    print('LAUNCHING STARMAP')
    if len(iterable)>0:
      process_count=min(subprocess_count,len(iterable))
      chunks = int(len(iterable)/process_count)  
      with get_context("spawn").Pool(processes=process_count, maxtasksperchild=1) as pool:
        results=pool.starmap(run_solve,iterable,chunksize=chunks)
      resolve_list = []
      for r in results:
        hdrnew = r[0]
        z = r[1]
        passed_solve_field = r[2]
        if not passed_solve_field:
          resolve_list.append(z)
        headersnew[z] = hdrnew  
      scale = min(imgshape[0], imgshape[1])
      #Report this initial run 
      print('STARMAP END',flush=True)
      subtime-=time.time()
      subtime/=60.
      print('Runtime: %.2f'%subtime)
      subtime=time.time()
      len_calibrate_detections = np.copy(len(iterable) - len(resolve_list)) #len first pass calibrate
      print('PASS1: iterable %d, resolve %d, result %d'%(len(iterable), len(resolve_list), len_calibrate_detections))

      #TODO: The followisn should become default behabior 
      #WHEN STARS WERE PYCIS BEFORE, NOW TRY SOURCE ON REDUCED SET 
      if imgastro==3:
        iterlen1 = np.copy(len(iterable) - len(resolve_list)) #len first pass calibrate
        print('RUNNING SOURCEEXTRACTOR FOR STARSOLVE ON %d/%d TRIALS'%(len(resolve_list),len(iterable)),flush=True)
        iterable,headersnew = run_astrometry_sub(resolve_list, folder, savename,goodlines,badlines,headers,headersnew,usetle,imgshape,input_dirs,maskset,extrap,
            jsonname,skipifexist,useimg,2,img,frame_zero,imscale,binfactor,importcorr,cropscale,iers_a)
        print('SECOND RUN: ITERABLE IS LENGTH: ',len(iterable))
        subtime-=time.time()
        subtime/=60.
        print('Runtime: %.2f'%subtime)
        subtime=time.time()
        print('LAUNCHING STARMAP')
        if len(iterable)>0:
          process_count=min(subprocess_count,len(iterable))
          chunks = int(len(iterable)/process_count)  
          with get_context("spawn").Pool(processes=process_count, maxtasksperchild=1) as pool:
            results=pool.starmap(run_solve,iterable,chunksize=chunks)
          resolve_list = []
          for r in results:
            hdrnew = r[0]
            z = r[1]
            passed_solve_field = r[2]
            if not passed_solve_field:
              resolve_list.append(z)
            headersnew[z] = hdrnew  
          scale = min(imgshape[0], imgshape[1])
          subtime-=time.time()
          subtime/=60.
          print('Runtime: %.2f'%subtime,flush=True)
          subtime=time.time()
          iterlen2 = np.copy(len(iterable) - len(resolve_list)) #len second pass calibrate 
          len_calibrate_detections = iterlen1+iterlen2
          print('PASS2: iterable %d, resolve %d, result %d'%(len(iterable), len(resolve_list), iterlen2))
          print('NET: ',len_calibrate_detections)

      #If only wanting XY results, report now 
      jsondata=[]
      if usetle==-1:
        for z in range(len(headersnew)):
          hdr = headersnew[z]
          if len(hdr)>0:
            num_obj = len(hdr)
            
            for obj in range(num_obj):
                rowdict={
                  'frame':z, 'num_objects':num_obj, 
                  'object_coords': [hdr["X"][obj],hdr["Y"][obj]], 'object_err': 0,
                  'scale':scale, 'units':'pix' }
                jsondata.append(rowdict)  
          else:
            rowdict={
            'frame':z, 'num_objects':0, 
            'object_coords': [], 'object_err': 0,
            'scale':scale, 'units':'pix' }
            jsondata.append(rowdict)  
        if makejson==1:
          jsonname='%s/record_%s.json'%(folder,savename)
          with open(jsonname,'w') as json_file:
            json.dump(jsondata,json_file)
        cleanup(folder)
        return headersnew

      #Otherwise, need to format output from the updated headersnew data 
      try:
        Lscale = hdr['CDELT1']
        Hscale = hdr['CDELT2']
        RAscale = 15.     
        if not (":" in str(hdr['CMND_RA'])):
          hdr['CMND_RA'] = convDMS(hdr['CMND_RA']/RAscale)
          hdr['CMND_DEC'] = convDMS(hdr['CMND_DEC'])
          print('updated header')
        else:
          hdr['CMND_RA'] = hdr['CMND_RA']/RAscale
      except: 
        RAscale=15.
    
      track = 0
      #IF JSON EXISTS
      #jsonname='%s/record_%s.json'%(folder,savename)
      if os.path.exists(jsonname) and skipifexist:
        with open(jsonname) as jsontemp:
            jsondata = json.load(jsontemp)
          
      #Iterate over each header and read in data 
      for z in range(len(headersnew)):
        hdr = headersnew[z]
        knownframes=[]
        if hdr['NUM_PY']>0:
            track+=1

            for i in range(hdr['NUM_PY']):
                try:
                  ra0 = convDD(hdr['RA_OBJ'])*RAscale
                  dec0 = convDD(hdr['DEC_OBJ'])
                except:
                  try:
                    ra0 = convDD(hdr['CMND_RA'])*RAscale #prior right ascension, degrees
                    dec0 = convDD(hdr['CMND_DEC']) #prior right assension, degrees
                  except:
                    ra0 = hdr['CMND_RA']*RAscale #prior right ascension, degrees
                    dec0 = hdr['CMND_DEC'] #prior right assension, degrees
                ra0 = convDD(hdr['R0_PY'])*RAscale #prior right ascension, degrees
                dec0 = convDD(hdr['D0_PY']) #prior right assension, degrees

                #Get TLE projection (TEME->GCRS) if provided
                if not tle:
                    pass
                else:
                    location = EarthLocation(lat=hdr['LAT_PY']*u.degree,
                        lon=hdr['LON_PY']*u.degree,height=hdr['ALT_PY']*u.meter)
                    satellite=Satrec.twoline2rv(tle[0],tle[1])
                    sattime = Time(hdr['TIME_PY'],format='fits')
                    sattime.format='jd'

                    error_code,teme_p,teme_v = satellite.sgp4(sattime.jd1,sattime.jd2)
                    if error_code !=0:
                        raise RuntimeError(SGP4_ERRORS[error_code])
                    teme_p = CartesianRepresentation(teme_p*u.km)
                    teme_v = CartesianDifferential(teme_v*u.km/u.s)

                    c_teme = TEME(teme_p.with_differentials(teme_v),obstime=sattime)
                    c_gcrs = c_teme.transform_to(GCRS(obstime=sattime))
                    ra0 = c_gcrs.ra.value
                    dec0 = c_gcrs.dec.value 

                # Get angles info 
                skycoord_0 = SkyCoord(ra0*u.deg,dec0*u.deg,frame='fk5') 
                ra0=ra0*np.pi/180.
                dec0=dec0*np.pi/180.

                ra = convDD(hdr['RA_PY%d'%i])
                dec = convDD(hdr['DEC_PY%d'%i])
                iloc = hdr['IDX_PY%d'%i]
                sloc = hdr['STR_PY%d'%i]

                skycoord = SkyCoord(ra*u.deg,dec*u.deg,frame='fk5') 
                ra *= RAscale*np.pi/180.
                dec *=np.pi/180.

                #This is only relevant is we consider a known TLE...
                err = np.arccos(np.cos(dec)*np.cos(dec0)+np.sin(dec)*np.sin(dec0)*np.cos(ra-ra0))
                errformat = convDMS(err*180./np.pi) #this is in rads but need degrees
                scale=0
                try:
                    naxis = min(hdr['NAXIS1'],hdr['NAXIS2'])
                    dpp = 1.76 / naxis 
                    errnew = (err / (dpp * np.pi/180)) /    naxis
                    scale = 1.76
                except:
                    errnew = (err / (hdr['CDELT1'] * np.pi/180)) /    (min(hdr['NAXIS1'],hdr['NAXIS2']))
                    scale = min(hdr['NAXIS1'],hdr['NAXIS2'])*hdr['CDELT1']

                #Add entry of detection to json structure , for later TDM conversion 
                rowdict={
                  'frame':int(z), 'num_objects':int(hdr['NUM_PY']), 'index':int(iloc), 'streaklike':int(sloc),
                  'object_coords': [float(ra), float(dec)], 'object_err': float(errnew), 
                  'scale':float(scale), 'units':'rads', 'time':str(hdr['TIME_PY']),
                  'x':int(np.ceil(hdr['X_PY%d'%i])), 'y':int(np.ceil(hdr['Y_PY%d'%i])),
                  'xcal':int(np.ceil(hdr['X0_PY%d'%i])), 'ycal':int(np.ceil(hdr['Y0_PY%d'%i])) }
                jsondata.append(rowdict)  
                printdata.append([int(np.ceil(hdr['X_PY%d'%i])),int(np.ceil(hdr['Y_PY%d'%i])),int(z)])
        else:
            pass

      ##SAVE JSON DATA 
      #print(jsondata)
      #print(json.dumps(jsondata,indent=4))
      if makejson==1:
        if not(os.path.exists(jsonname) and skipifexist):
          jsonname='%s/record_%s.json'%(folder,savename)
          #jsonname='%s/SE1_record_%s.json'%(folder,savename)
          print('WRITING RECORD TO %s'%jsonname)
          with open(jsonname,'w') as json_file:
              json.dump(jsondata,json_file)
      cleanup(folder)
      #print('SUCCESSFUL ASTROMETRY ON %d/%d (%.0f)'%(track,len(headersnew),track/len(headersnew)*100.))


      if not (os.path.exists(jsonname) and skipifexist):
          #print('SUCCESSFUL ASTROMETRY:')
          #print('\t calibrated detections / all files:      %d/%d (%.0f)'%(track,len(headersnew),track/len(headersnew)*100.))
          #print('\t calibrated detections / all detections: %d/%d (%.0f)'%(track,iterlen,track/iterlen*100.))
          #print('\t        all detections / all files       %d/%d (%.0f)'%(iterlen,len(headersnew),iterlen/len(headersnew)*100.))
          track=len_calibrate_detections
          iterlen=len_all_detections
          print('SUCCESSFUL ASTROMETRY:')
          print('\t calibrated detections / all files:      %d/%d (%.0f)'%(track,len(headersnew),track/len(headersnew)*100.))
          print('\t calibrated detections / all detections: %d/%d (%.0f)'%(track,iterlen,track/iterlen*100.))
          print('\t        all detections / all files       %d/%d (%.0f)'%(iterlen,len(headersnew),iterlen/len(headersnew)*100.))

      ##CONVERT JSON TO A CCSDS-TDM STRUCTURE AND SAVE PLOTTING DATA
      #tdmname='%s/record_%s.tdm'%(tdmfolder,savename)
      #tdmname='%s/record_%s.xml'%(tdmfolder,savename)
      tdmname='%s/%s.xml'%(tdmfolder,savename)
      

      printdata,rdtdata,title=convert_json_tdm(jsondata,headersnew,tdmname,tol=errtol)

      #SAVE THE NEW PLOT RESULTS 
      print_detections_window_postastro(imgshape,printdata,goodlines,folder=folder,savename='ASTRO_%s'%savename,
        makevid=0,makegif=1,vs=vs,fps=5,amp=1,background=0,
        ccopt=len(headers), I3loc = I3loc, frames=frames,
        imscale=imscale, median=median, shift=shift, datatype=datatype,binfactor=binfactor,rdt=rdtdata,title=title,starlines=badlines)

      #TODO: Clean up this end statment, should immediatly end after writing.  Remove while loop and track
      if track>0:
          #print(json.dumps(jsondata,indent=4))
          useimg+=2
      else:
          print('FAIL')
          #quit()
          #useimg+=1
          useimg+=2
  return headersnew
