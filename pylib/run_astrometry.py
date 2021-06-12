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
*Date of Modification: June 07, 2021

**NOTICE: For copyright and licensing, see 'notices' at bottom of README
'''

import os
import glob
import subprocess
import numpy as np
from pylib.print_detections import interp_frame

#from astropy import _erfa as erfa
import erfa
from astropy.utils.iers import conf, LeapSeconds
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation

def load_astropy():
  ''' load updates and/or disable auto-update good for parallelization '''
  #Update should be updated once at start
  conf.auto_download = True
  conf.remote_timeout=5
  conf.reload('remote_timeout')
  conf.reload('auto_download')
  table = LeapSeconds.auto_open()
  erfa.leap_seconds.update(table)
  #Should thereafter always be updated to turn off 
  conf.auto_download = False
  conf.remote_timeout=0
  conf.reload('remote_timeout')
  conf.reload('auto_download')

def interp_frame_xy(locline,z, star=0):
    ''' Get x,y line to draw on z-level of video, accound for single-frame streaks '''
    x1,y1,x2,y2 = interp_frame(locline,z)
    z1 = int(locline[2]); z2 = int(locline[5]); 
    if (star==1) and not ((z1==z2) and (z1==z)):
        return 0,0
    if (z1==z2):
      x = (x1+x2)/2.; y = (y1+y2)/2.; 
    else:
      x = x2; y = y2; 
    return x,y

def convDMS(x,hours=0):
    ''' convert DD angles to DMS (dd:mm:ss.sss) (if hh:mm:ss, use DD=convDD(x)/15.) '''
    sign = x < 0
    x = abs(x)
    mm,ss= divmod(x*3600,60)
    dd,mm = divmod(mm,60)
    sa=int(ss)
    sb = int((ss-sa)*1000)
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

def make_xyls(ptslist, hdr, name, obj=0):
  ''' 
  Save xyls table of point data
  As temporary handling of linearity constraint, 
  group obj detections within a given tolerance weighed by NFA
  '''
  #ensure dimension
  if (ptslist.ndim==1):
    np.expand_dims(ptslist,axis=0)
  lcount = len(ptslist)
  data = Table()
  
  #for objects, we apply a correction due to the linearity constraint, 
  #averaging detections which occur within some tolerance
  if (obj==1) and (lcount>1):
    tol = hdr['XVISSIZE']*.01 #pixel tolerance
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
    data['X'] = ptslist[:,0] 
    data['Y'] = ptslist[:,1]
    data['NFA'] = ptslist[:,2]
  data.write('%s.xyls'%name,overwrite=True,format='fits')

def solve_field(name,hdr):
  ''' 
  solve name.wcs from name.xyls star list 
  We use the 2MASS and Tycho-2 star catalogs at 30-2000 arcminute diameters.
  Loading additional 2MASS index files enables stronger solving at higher storage cost

  '''
  w = hdr['XVISSIZE'] #field width, pixels
  e = hdr['YVISSIZE'] #field height, pixels
  L = 1.75 #lower bound of image width, degrees
  H = 1.77 #upper bound of image width, degrees
  ra = convDD(hdr['RA'])*15. #prior right ascension, degrees
  dec = convDD(hdr['DEC']) #prior right assension, degrees
  radius = H*2. #search radius, degrees

  #run solve, permitting error bounds on solve (centerline length innacuracies)
  if os.path.exists('%s.solved'%name):
    os.remove('%s.solved'%name)
  passer=0
  tol=1
  while passer==0:
    solve_command = "solve-field --no-plots  --no-verify --overwrite -E %f "%(tol)
    solve_command+= "-X X -Y Y -s NFA -w %d -e %d -u dw -L %f -H %f --ra %f --dec %f --radius %f %s"%(
      w, e, L, H, ra, dec, radius, '%s.xyls'%name)
    solve_result = subprocess.run(solve_command,shell=True, 
      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL )
    if os.path.exists('%s.solved'%name):
      passer=1
    else:
        if tol==1:
            tol+=1
        else:
            tol*=tol
        if tol>w*.01: #at once percent
            passer=1




def convert(starname, objname, hdrin, frame):
  ''' convert object data from pixel to radec coords using stellar astrometry '''
  convert_command = 'wcs-xy2rd -w %s -i %s -o %s'%('%s.wcs'%starname, '%s.xyls'%objname, '%s.rdls'%objname)
  convert_result = subprocess.run(convert_command, shell=True) #convert rso data
  #update header
  hdr = hdrin.copy()
  hdulrd = fits.open('%s.rdls'%objname)
  datard = hdulrd[1].data
  hdulrd.close()
  hdulxy = fits.open('%s.xyls'%objname)
  dataxy = hdulxy[1].data
  hdulxy.close()

  numrows = 0

  for i in range(len(datard)):
    ra =  datard[i].field(0)
    dec=  datard[i].field(1)
    '''
    per http://data.astrometry.net/4200 :
    solutions are by default in  the J2000.0 ICRS reference system.
    We convert celestial solutions to FK5 for header agreement
    We convert apparent  solutions to TEME for prior agreement
    '''
    #Load ICRS coordinates, telescope location, and time with exposure offset
    c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs',
            obstime=Time(hdr['DATE-OBS'],format='isot',scale='utc')+
              TimeDelta(hdr['EXPOSURE']/2./86400.),
            location = EarthLocation(lat=hdr['TELLAT']*u.degree,
                lon=hdr['TELLONG']*u.degree,height=hdr['TELALT']*u.meter),
            )
    
    c_fk5 = c_icrs.transform_to('fk5')
    ra_fk5 = c_fk5.ra.value
    dec_fk5 = c_fk5.dec.value

    c_teme = c_icrs.transform_to('teme')
    c_teme.representation_type='spherical'
    ra_teme = c_teme.lon.value
    dec_teme = c_teme.lat.value


    hdr.set('RA_PY%d'%i, convDMS(ra_fk5/15.), 'PyCIS target RA FK5 [hour]')
    hdr.set('DEC_PY%d'%i, convDMS(dec_fk5), 'PyCIS target Dec FK5 [deg]')
    hdr.set('ARA_PY%d'%i, convDMS(ra_teme/15.), 'PyCIS target RA apparent TEME [hour]')
    hdr.set('ADEC_PY%d'%i, convDMS(dec_teme), 'PyCIS target Dec apparent TEME [deg]' )

    hdr.set('NFA_PY%d'%i, dataxy[i].field(2), 'PyCIS centerline NFA')
    numrows+=1
    hdr.set('NUM_PY', numrows, 'PyCIS number of detections')
  return hdr

def null_hdr(hdrin):
  ''' update header with numpy=0 '''
  hdr = hdrin.copy()
  hdr.set('NUM_PY', 0, 'PyCIS number of detections')
  return hdr

def cleanup(folder):
  ''' remove temp files'''
  imlistwcs = glob.glob('%s/*.wcs'%(folder))
  imlistaxy = glob.glob('%s/*.axy'%(folder))
  imlistcorr = glob.glob('%s/*.corr'%(folder))
  imlistmatch = glob.glob('%s/*.match'%(folder))
  imlistrdls = glob.glob('%s/*.rdls'%(folder))
  imlistsolved = glob.glob('%s/*.solved'%(folder))
  for im in imlistwcs:
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

def run_astrometry(goodlines, badlines, headers, folder='temp',savename='temp'):
  '''
  Launch Astrometry.net plate solving, group and convert a-contrario detections, 
  and update headers for updating fits files or JSON output
  Input: 
    goodlines -  line features corresponding to 2nd-order-meaninful detected objects 
    badlines -   line features corresponding to 1st-order-meaninful stars and noise
    headers -    original headers from fits files
  Output: 
    headersnew - updated headers with detection ra/dec and nfa information
  '''

  load_astropy()

  #Interpolate each frame
  headersnew = []
  for z in range(len(headers)):
    print('Solving frame ',z)

    starname='%s/%s_star%d'%(folder,savename,z)
    objname='%s/%s_obj%d'%(folder,savename,z)
    hdr = headers[z]

    
    if (len(badlines)==0) or (len(goodlines)==0):
      headersnew.append(null_hdr(hdr))
      continue

    #create star list
    starlist = []
    for k in range(len(badlines)):
      locline = badlines[k,:6].squeeze()
      x,y= interp_frame_xy(locline,z,star=1)
      if not all(np.array([x,y])==0):
        starlist.append([x,y,badlines[k,-1]])


    #create obj list
    objlist = []
    for k in range(len(goodlines)):
      locline = goodlines[k,:6].squeeze()
      x,y = interp_frame_xy(locline,z)
      if not all(np.array([x,y])==0):
        objlist.append([x,y,goodlines[k,-1]])


    #account for no-detection case
    if (len(starlist)==0) or (len(objlist)==0):
      headersnew.append(null_hdr(hdr))
      continue

    #save astrometry-friently fits tables
    make_xyls(np.array(starlist),hdr, starname)
    make_xyls(np.array(objlist), hdr, objname, obj=1)

    #solve plate
    solve_field(starname, hdr)
    if not os.path.exists('%s.solved'%starname):
        print('failed, continuting')
        headersnew.append(null_hdr(hdr))
        continue

    #solve stars and update header
    hdrnew = convert(starname, objname, hdr, z)
    headersnew.append( hdrnew )


  #Print track log
  track = 0
  for z in range(len(headersnew)):
    hdr = headersnew[z]
    print('FRAME=%d'%z)
    print('Prior: (',hdr['RA_OBJ'],', ',hdr['DEC_OBJ'],')')
    if hdr['NUM_PY']>0:
        track+=1
        for i in range(hdr['NUM_PY']):
            ra0 = convDD(hdr['RA_OBJ'])*15.*np.pi/180.
            dec0 = convDD(hdr['DEC_OBJ'])*np.pi/180.
            ra = convDD(hdr['RA_PY%d'%i])*15.*np.pi/180.
            dec = convDD(hdr['DEC_PY%d'%i])*np.pi/180.
            err = np.arccos(np.cos(dec)*np.cos(dec0)+np.sin(dec)*np.sin(dec0)*np.cos(ra-ra0))
            print('Post : (',hdr['RA_PY%d'%i],', ',hdr['DEC_PY%d'%i],'), err: ',convDMS(err))
  cleanup(folder)
  print('SUCCESSFUL ASTROMETRY ON %d/%d (%.0f)'%(track,len(headersnew),track/len(headersnew)*100.))

  return headersnew