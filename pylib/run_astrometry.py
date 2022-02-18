'''
PyCIS - Python Computational Inference from Structure

pylib/run_astrometry.py: Generate right ascension - declination solution.
  Optionally defines error relative to an expected TLE track (under development).

TODO:
  Format output to CCSDS Tracking Data Message (TDM) format. 
  Use CAST-local astrometry solution software in place of Astrometry.net offline pipline 

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
import json
import glob
import subprocess
import numpy as np
from pylib.import_fits import import_fits
from pylib.print_detections import interp_frame, interp_frame_xy, make_kernel
from pylib.detect_outliers import detect_outliers
from datetime import datetime
from astropy import _erfa as erfa
#import erfa
from astropy.utils.data import clear_download_cache
from astropy.utils import iers
from astropy.utils.iers import conf, LeapSeconds
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.coordinates import TEME, ICRS, ITRS, GCRS, FK5, CIRS, AltAz
from astropy.coordinates import CartesianDifferential,CartesianRepresentation
from multiprocessing import Pool, cpu_count,get_context,set_start_method
from scipy.ndimage import convolve,maximum_filter, gaussian_filter
import cv2
import imageio
from time import sleep

import skyfield.sgp4lib as sgp4lib
from sgp4.api import Satrec
from sgp4.api import SGP4_ERRORS
import sep
import pandas as pd
def fill_data_leg(array,order,robust=False):
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
        #mask =~(np.abs(diff-np.mean(diff)) > np.std(diff))
        diff = np.abs(diff)
        print('input')
        print(array)
        print('filter')
        print(array2)
        print('filter diff')
        print(diff)
        diff = (diff - np.mean(diff))/np.std(diff)
        mask = ~(diff > np.quantile(diff,0.75))
        #mask =~(np.abs(diff-np.mean(diff)) > np.std(diff))
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
        coeff = np.polynomial.legendre.legfit(idx2,array2,int(order),w=weight)
    else:
        coeff = np.polynomial.legendre.legfit(idx2,array2,int(order))

    xfit =  np.polynomial.legendre.legval(idx,coeff)
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

def make_xyls(ptslist, hdr, name, obj=0,img=[],imgloc="",frame_zero=-1,
  z=0, shape=[0,0],imscale=1,binfactor=1,importcorr=0,kernel=[]):

  ''' 
  Save xyls table of point data.
  As temporary handling of linearity constraint, 
  group obj detections within a given tolerance weighed by NFA.
  HARDCODED MANUAL GROUND TRUTH LABEL
  '''
  #ensure dimension
  if (ptslist.ndim==1):
    np.expand_dims(ptslist,axis=0)
  lcount = len(ptslist)
  data = Table()
  
  #for objects, we apply a correction due to the linearity constraint, 
  #averaging detections which occur within some tolerance
  if (obj==1) and (lcount>1):
    #try:
    #  tol = hdr['XVISSIZE']*.01 #pixel tolerance
    #except:
    #  tol = hdr['NAXIS1']*.01 #pixel tolerance
    tol = min(shape)*.01;
    inclist=[]
    ptstemp2 = []
    for i in range(lcount):
      if i not in inclist:
        ptstemp1 = []
        ptstemp1.append(ptslist[i])
        inclist.append(i)
        for j in range(i,lcount):
          if (i<j) and (j not in inclist):
            dist=((ptslist[i,0]-ptslist[j,0])**2.+(ptslist[i,1]-ptslist[j,1])**2.)**.5
            if dist<tol:
              ptstemp1.append(ptslist[j])
              inclist.append(j)

        #take mean of nearby data and add to list
        ptstemp1 = np.array(ptstemp1)
        if (ptstemp1.ndim==1):
          np.expand_dims(ptstemp1,axis=0)
        #weighted mean according to nfa
        weights = ptstemp1[:,2] / sum(ptstemp1[:,2])
        ptstemp1 = np.sum(ptstemp1*np.expand_dims(weights,axis=1),axis=0)
        ptstemp2.append(ptstemp1)

    ptstemp2=np.array(ptstemp2)
    if (ptstemp2.ndim==1):
      np.expand_dims(ptstemp2,axis=0)

    #create data table
    data['X'] = ptstemp2[:,0] 
    data['Y'] = ptstemp2[:,1]
    data['NFA'] = ptstemp2[:,2]

  else:
    #otherwise use provided data OR estimate from image  
    if (len(img)==0) and (not imgloc):
        data['X'] = ptslist[:,0] 
        data['Y'] = ptslist[:,1]
        data['NFA'] = ptslist[:,2]
        #data['OBJ'] = ptslist[:,3]
        pass
    else:#maximize 
        datatmp=[]
        #Solver uses basic source extraction 
        #should resort to this if star aliasing is too severe 
        importlist=[]
        if imgloc:
            if importcorr==1 or importcorr==0 or importcorr==-1:
                frame_z = int(frame_zero+z+importcorr)
                print('importing frame ',frame_z)
                img, _ = import_fits(imgloc, savedata=0, subsample=1,
                    framerange=[frame_z,frame_z], scale=imscale,binfactor=binfactor)
                sub = img.squeeze()
            elif importcorr==3:
                for importcorrloc in [-1,0,1]:
                    frame_z = int(frame_zero+z+importcorr)
                    print('importing frame ',frame_z)
                    img, _ = import_fits(imgloc, savedata=0, subsample=1,
                        framerange=[frame_z,frame_z], scale=imscale,binfactor=binfactor)
                    importlist.append(img.squeeze())

            else:
                frame_z = int(frame_zero+z)
                print('importing frame ',frame_z)
                img, _ = import_fits(imgloc, savedata=0, subsample=1,
                    framerange=[frame_z,frame_z], scale=imscale,binfactor=binfactor)
                sub = img.squeeze()
        else:
            sub = np.copy(img[:,:,int(z)].squeeze())
        if not importlist:
            print('',flush=True)
            if len(kernel)==0:
                sub = gaussian_filter(sub,sigma=2)
                bkg= sep.Background(sub)
                sub -= bkg
                objs = sep.extract(sub,3, err = bkg.globalrms)
            else:
                #sub = gaussian_filter(sub,sigma=2)
                #objs = sep.extract(sub,3.0, err = bkg.globalrms, filter_kernel=kernel, filter_type='conv')

                #subtract before convolve - need to ensure we dont explode
                #bkg= sep.Background(sub) #recompute background and resolve
                #sub -= bkg
                #sub = sub-np.amin(sub)
                #convolve and resubtract
                sub = convolve(sub,kernel)
                bkg= sep.Background(sub) #recompute background and resolve
                sub -= bkg
                sub = sub-np.amin(sub)
                print('submin ',np.amin(sub))

                savemean = np.mean(sub)
                savestd = bkg.globalrms
                teststd = 3.0
                sizethresh = 0.005 #5 percent max
                testimg = sub-savemean #no abs, seek only bright
                sizetest =  float(np.count_nonzero( testimg > (teststd*savestd)  )) / float(sub.size)
                print('thresh %d yeilds %.3f, compare to %.3f'%(int(teststd),sizetest,sizethresh),flush=True)
                while sizetest > sizethresh:
                    teststd=teststd+1.0
                    sizetest =  float(np.count_nonzero( testimg > (teststd*savestd)  )) / float(sub.size)
                    print('thresh %d yeilds %.3f, compare to %.3f'%(int(teststd),sizetest,sizethresh),flush=True)


                #objs = sep.extract(sub,teststd, err = bkg.globalrms, filter_kernel=kernel, filter_type='matched')

                objs = sep.extract(sub,teststd, err = bkg.globalrms)
                #print('saveing img %s.png'%name)
                #saveimg = np.ceil(sub*(254./np.amax(sub))).astype(np.uint8)
                #print('submax ',np.amax(saveimg))
                #imageio.imwrite('%s.png'%name,saveimg)


        ptstemp = np.empty((len(objs),3))
        for i in range(len(objs)):

            ptstemp[i,0] = objs['x'][i] 
            ptstemp[i,1] = objs['y'][i]
            ptstemp[i,2] = objs['peak'][i]
              
        data['X'] = ptstemp[:,0] 
        data['Y'] = ptstemp[:,1]
        data['NFA'] = ptstemp[:,2]
        #filtering for admissibility - currently unimplemented
        tempfilt = np.copy(np.asarray(data['X'][:]))>0
        datanew = Table()
        datanew['NFA']=data['NFA'][tempfilt]
        datanew['X']=data['X'][tempfilt]
        datanew['Y']=data['Y'][tempfilt]
        del data
        data=datanew

  data.write('%s.xyls'%name,overwrite=True,format='fits')

  #Save manual label ('calibration') 
  #HARDCODED LABEL REQUIRES UPDATE FOR FUTURE TESTS 
  datacal = Table()
  datacal['NFA']=[10001,]
  datacal['Y']=[shape[0]/2]
  datacal['X']=[shape[1]/2]
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
  try:
    Lscale = 0.8*hdr['CDELT1']*hdr['NAXIS1'] #deg/pix * pix = FOV
    Hscale = 1.2*hdr['CDELT2']*hdr['NAXIS2']
    RAscale=1.
  except: 
    RAscale=15.
  try: #FOV*(current length (unbinned)* length) = newFOV
    L = Lscale*(w*binfactor/hdr['XVISSIZE']) #lower bound of image width, degrees
    H = Hscale*(e*binfactor/hdr['YVISSIZE']) #upper bound of image width, degrees
  except:
    L = Lscale*(w*binfactor/hdr['NAXIS1']) #lower bound of image width, degrees
    H = Hscale*(e*binfactor/hdr['NAXIS2']) #upper bound of image width, degrees
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
  #print('w,e,l,h,ra,dec : ',[w,e,L,H,ra,dec])
  #run solve, permitting error bounds on solve 
  if os.path.exists('%s.solved'%name):
    os.remove('%s.solved'%name)
  passer=0
  tol=1
  while passer==0: #no-remove-lines
    solve_command = "solve-field --no-plots  --no-verify --no-remove-lines --overwrite -E %f  -d 20,50,100 "%(tol)
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
    if os.path.exists('%s.solved'%name):
      passer=1
    else:
        if tol==1:
            tol+=1
        else:
            tol*=tol
        if tol>min(w*.01,20): #at once percent
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
    c_fk5 = c_icrs.transform_to('fk5')
    ra_fk5 = c_fk5.ra.value
    dec_fk5 = c_fk5.dec.value
    c_teme = c_icrs.transform_to('teme')
    c_teme.representation_type='spherical'
    ra_teme = c_teme.lon.value
    dec_teme = c_teme.lat.value
  
    #IF USING TLE, CONVERT FROM J2000 PROJECTION TO GCRS LOS CORRECTION 
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
            print('z %d failed, continuting'%(z),flush=True)
            return null_hdr(hdr,usetle), z
        else:
            print('z %d passed, continuting'%(z),flush=True)
        #solve stars and update header
        #print('pass successful')
        hdrnew = convert(starname, objname, hdr, z, usetle)
    else:
        print('z %d returning XY'%(z),flush=True)
        hdrnew = convertXYONLY(starname, objname, hdr, z, usetle)
    return hdrnew,z


def build_kernels(hlen,goodlines,badlines,input_dirs):
  maskset = np.nan*np.ones((hlen,3))
  if 1==1:
      for z in range(hlen):
        print('adding z %d'%z, flush=True)
        if (len(badlines)==0) or (len(goodlines)==0):
          continue

        starlist = []
        tempstarlines=[]
        for k in range(len(badlines)):
          locline2 = badlines[k,:].squeeze()
          locline = badlines[k,:6].squeeze()
          x,y= interp_frame_xy(locline,z,star=0)
          if not all(np.array([x,y])==0):
            tempstarlines.append(locline2)
            val = badlines[k,-1]
            starlist.append([x,y,badlines[k,-1],0])
        if not input_dirs:
            pass
        else:
            lines = np.vstack(tempstarlines)
            k_len= np.linalg.norm(lines[:,:3]-lines[:,3:6],axis=1)
            el = np.arccos((lines[:,5]-lines[:,2])/k_len)
            az = np.arctan2((lines[:,3]-lines[:,0]) , (lines[:,4]-lines[:,1]))

            elmask = (el*180./np.pi) > (90.-22.5)
            print('considering %d/%d non-ASO lines'%(np.count_nonzero(elmask),len(elmask)))
            if np.count_nonzero(elmask)>1:
                medlen = np.median(k_len[elmask])
                kmask = k_len>medlen
                elmask = np.logical_and(elmask,kmask)
                print('update: considering %d/%d non-ASO lines'%(np.count_nonzero(elmask),len(elmask)))
                if np.count_nonzero(elmask)>1:
                    medaz = np.median(az[elmask])
                    medwid = np.median(lines[elmask,6])
                    print('kernel length, width, az: ',[medlen,medwid,medaz*180./np.pi])
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

def run_astrometry(img,goodlines, badlines, headers, scale=1., folder='temp',savename='temp',
  makejson=0,tle=[],binfactor=1,imgastro=0,subprocess_count=10,
   I3loc=[],frames=[],
  imscale=1.,median=1,shift=0,datatype='fits',kernel=[]):
  '''
  Launch Astrometry.net plate solving, group and convert a-contrario detections, 
  and update headers for updating fits files or JSON output
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
  '''
  cropscale=np.copy(scale)
  #for importcorr in [-1,0,1]:
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

  maskset = build_kernels(len(headers),goodlines,badlines,input_dirs)

  while useimg<=1:
      for z in range(len(headers)):
        starname='%s/%s_star%d'%(folder,savename,z)
        objname='%s/%s_obj%d'%(folder,savename,z)
        hdr = headers[z]
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
          x,y= interp_frame_xy(locline,z,star=0)
          if not all(np.array([x,y])==0):
            tempstarlines.append(locline2)
            val = badlines[k,-1]
            starlist.append([x,y,badlines[k,-1],0])
        if not input_dirs:
            pass
        else:
            lines = np.vstack(tempstarlines)

            medlen = maskset[int(z),0]
            medwid = maskset[int(z),1]
            medaz = maskset[int(z),2]
            if (not np.isnan(medlen)):
                kernel = make_kernel(medlen,medwid,medaz)

        #create obj list
        objlist = []
        for k in range(len(goodlines)):
          locline = goodlines[k,:6].squeeze()
          x,y = interp_frame_xy(locline,z)
          if not all(np.array([x,y])==0):
            objlist.append([x,y,goodlines[k,-1]])
            print('obj',[x,y,z,k+1])
        #'''
        #account for no-detection case
        if (len(starlist)==0) or (len(objlist)==0):
          headersnew[z] =null_hdr(hdr,usetle)
          continue

        if usetle==-1 or (useimg==0 and imgastro==0):
          make_xyls(np.array(starlist),hdr, starname, z=z, shape=img.shape)#, img=img,z=z)
        else:
          if not input_dirs:
              make_xyls(np.array(starlist),hdr, starname, img=img,z=z, shape=img.shape)
          else:
              #print("SHOULD BE USING IMPORT",flush=True)
              make_xyls(np.array(starlist),hdr, starname, imgloc=input_dirs[0],frame_zero=frame_zero,z=z, shape=img.shape,imscale=imscale,binfactor=binfactor,importcorr=importcorr,kernel=kernel)
        #print('DISABLING OBJ INTERPOLATION SINCE WE NOW HAVE ASSOC STEP')
        #print('OBJS num',len(objlist))
        #print(objlist)
        make_xyls(np.array(objlist), hdr, objname, obj=0, z=z, shape=img.shape)
        iterable.append((starname,objname,hdr,cropscale,z,iers_a,binfactor,usetle))
      print('ITERABLE IS LENGTH: ',len(iterable))
      iterlen = len(iterable)


      ## RETURN ONLY XY IF REQUESTED
      if len(iterable)>0:
        process_count=min(subprocess_count,len(iterable))
        chunks = int(len(iterable)/process_count)  
        with get_context("spawn").Pool(processes=process_count, maxtasksperchild=1) as pool:
          results=pool.starmap(run_solve,iterable,chunksize=chunks)
        for r in results:
          hdrnew = r[0]
          z = r[1]
          headersnew[z] = hdrnew  
        jsondata=[]
        scale = min(img.shape[0], img.shape[1])

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


      #Print track log
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
      jsondata=[]
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
                skycoord = SkyCoord(ra*u.deg,dec*u.deg,frame='fk5') 
                ra *= RAscale*np.pi/180.
                dec *=np.pi/180.
            
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

                rowdict={
                  'frame':z, 'num_objects':hdr['NUM_PY'], 
                  'object_coords': [ra, dec], 'object_err': errnew, 
                  'scale':scale, 'units':'rads' }
                jsondata.append(rowdict)  
        else:
            rowdict={
            'frame':z, 'num_objects':hdr['NUM_PY'], 
            'object_coords': [], 'object_err': 0, 
            'scale':0,'units':'rads'
            }
            jsondata.append(rowdict)  
      if makejson==1:
            jsonname='%s/record_%s.json'%(folder,savename)
            print('WRITING RECORD TO %s'%jsonname)
            with open(jsonname,'w') as json_file:
                json.dump(jsondata,json_file)

      cleanup(folder)
      #print('SUCCESSFUL ASTROMETRY ON %d/%d (%.0f)'%(track,len(headersnew),track/len(headersnew)*100.))
      print('SUCCESSFUL ASTROMETRY:')
      print('\t calibrated detections / all files:      %d/%d (%.0f)'%(track,len(headersnew),track/len(headersnew)*100.))
      print('\t calibrated detections / all detections: %d/%d (%.0f)'%(track,iterlen,track/iterlen*100.))
      print('\t        all detections / all files       %d/%d (%.0f)'%(iterlen,len(headersnew),iterlen/len(headersnew)*100.))

      if track>0:
          #print(json.dumps(jsondata,indent=4))
          useimg+=2
      else:
          print('FAIL')
          #quit()
          #useimg+=1
          useimg+=2
  return headersnew
