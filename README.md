# PyCIS : Python Computational Inference from Structure


    An a-contrario detection algorithm for space object tracking from optical time-series telescope data. 


Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
*Date of Modification: February 16, 2022*

**NOTICE**: For copyright and licensing, see bottom of readme

------------------------------------------------------------------

## OVERVIEW:

This software aims to enhance traditional techniques for space object detection by mitigating processing requirements such as background subtraction, directly leveraging temporal information, and avoiding contrast-depentent measurements. The method does not require training data, and may be used to propagate uncertainty from data aquisition to the orbit determination processes.  This software can be used on either sidereal or rate-tracked data as potential ASOs have anomalous behavior relative to the starfield and background noise in either context.  This detection method is not designed for photometric analysis.


Under the *a-contrario* paradigm, structures in data are considered "meaningful" if they are unlikely to occur by chance.  In contrast to standard p-significance hypothesis testing, the *a-contrario* approach controls the number of detections in expectation within the data rather than by controling a probability of detection. PyCIS uses a recursive sequence of *a-contrario* steps to detect potential ASOs without making any high-fidelity *a-priori* model of predicted objects or the complicated noise structures (atmospheric, hardware, or stellar noise).  The algorithm first finds trajectories of objects surprising in the general atmospheric/sensor noise, and then infers structures of star and hardware behaviors.  The final detection is the set of features which cannot be attributed to any other noise infered from the data.  


Astrometric positioning from the detected class of star features is provided by the offline Astrometry.Net software.  This enables output of time-tagged right asension/ declination Tracking Data Messeges of the line-of-sight vector from raw optical data for further processing.


A formal report on this software and preliminary performance results is pending.  


------------------------------------------------------------------

## OPERATION:

**Setup**:
   If scripts cannot be read, run:
   `<dos2unix *.sh>`
   Finally, install prerequisites and build the software by running:
   `<. setup.sh>`.
    The software installation should take under 20 minutes but may require 1-2hrs to gather the astrometric star catalog. 
   After installing the star catalog, size is 2.7G, 1.4G for Astrometry.net 

**Data Download**:
    Run `<. getdata.sh>` to  automatically download full time-series ASTRIANet data from the link below.
    This will take about 5 minutes and install 5GB of analysis data, bringing total memory usage to 7.7GB.

  Feuge-Miller, Benjamin; Kucharski, Daniel; Iyer, Shiva; Jah, Moriba, 2021, "ASTRIANet Data for: Python Computational Inference from Structure (PyCIS)", https://doi.org/10.18738/T8/GV0ASD, Texas Data Repository, V3  

**Demo**:
    After setup and data download, run (for both local and TACC machines):
   `<. rundemo.sh>`
   The demo will reset paths using setup as necessary.  

**Troubleshooting**: 
    The demo scripts most often fail due to memory restrictions.  Try reducing the 'framerange', 'scale', and 'subprocess' parameters in  `<runpycis.py>` if necessary.



------------------------------------------------------------------

## DEMO VISUAL OBSERVATION
We provide two examples which can be recovered by running `<. getdata.sh>` and `<. rundemo.sh>`.  Below are two GIF-format video visuals showing detections over the optical data frames, made after subtracting the median value of each pixel as a naive dark/bias correction to mitigate obvious sensor noise (e.g. hot pixels) .  We also provide HTML-format interactive plots allowing visualization of the data as a dense data cube.  For the HTML data, we remove very dim pixels to create a pixel point cloud across all frames, where blue-yellow color corresponds to pixel brightness.  The "potential ASO" detections are marked in red, and opacity is varied for visibility.   

Navstar-48 (Medium Earth Orbit) demo, unbinned raw data using tighter angular tolerance (factor 1/1.3) for 1st-order line detection.  Detections are plotted in red, and include both the actively tracked Navstar in the center of the frame and a serendipitous detection of an Unexpected Orbital Object in the lower right. The low tracking noise for Medium Earth Orbit (and higher orbit) objects enables strong performance.  See [Interactive Starlink Plot](https://rawcdn.githack.com/ut-astria/PyCIS/d1c7a943fb445505a5d87da48b07fb0446b5a90a/docs/a0t10m1b2_ASSOC1T60W20O0_e0_inj-1_20201220_45696_starlink-1422_FINALVOL.html) for further visualization. 
![Demo Video](docs/videoObj_a0t13m1b1_ASSOC1T80W20O0_e0_inj-1_20201224_26407_navstar-48.gif)

Starlink-1422 (Low Earth Orbit) demo, using 2-pixel binning.  The 21st-40th frames of the time-series.  The blue and yellow lines are all candidate features in the data including stars and atmospheric/hardware noise, where the colormap indicates the time frame of detection.  The red lines are "potential ASOs".  There is high tracking noise when following a Low Earth Orbit object results in false positives (stars) and false negatives (missing tracks), which we will address in future updates.  See  [Interactive Starlink Plot](https://rawcdn.githack.com/ut-astria/PyCIS/d1c7a943fb445505a5d87da48b07fb0446b5a90a/docs/a0t13m1b1_ASSOC1T80W20O0_e0_inj-1_20201224_26407_navstar-48_FINALVOL.html) for further visualization. 
![Demo Video](docs/videoAll_a0t10m1b2_A21B40_e0_inj-1_20201220_45696_starlink-1422.gif)

Running demo script takes approximatly 15 minutes to process the Starlink example (with 2-pixel binning) and 60 minutes to process the Navstar example (with 1-pixel binning).  This was tested on a single Skylake node of the Stampede2 system of the Texas Advanced Computing Center at the University of Texas at Austin.  

------------------------------------------------------------------

## DATASET

The datasets available at the Texas Data Repository link below contains a sequence of 66 FITS-format data frames collected from the 
New Mexico Skies telecope of the ASTRIANet telescope network, tracking the Starlink-1422 satellite on the night of Dec. 20, 2020. Relevant observation data can be found 
in the FITS headers, printable by turning on the 'headerflag' variable in 
`<import_fits.py>`.  See citations below.  

Feuge-Miller, Benjamin; Kucharski, Daniel; Iyer, Shiva; Jah, Moriba, 2021, 
"ASTRIANet Data for: Python Computational Inference from Structure (PyCIS)", 
https://doi.org/10.18738/T8/GV0ASD, Texas Data Repository, V3  

------------------------------------------------------------------

## DEPENDENCIES:
The following will be installed at setup.  See <setup.sh> and <requirements.txt> for more details. 
 
* Before Installing: 
    * Python 3.8
    * Cmake
    * GCC
* PyCIS-LSD software
    * Python3 environment and C libraries 
* Point alignment software
    * FLANN
    * 3D ponint alignment detector v1.0 by Alvaro Gomez: http://dx.doi.org/10.5201/ipol.2015.126
* Astrometry software
    * CFITSIO
    * WCSlib
    * Astrometry.net offline software

------------------------------------------------------------------

## TOC:

* demo.py -           run demo as discussed above
* rundemo.sh -        activate env and launch demo
* runcron.sh -        template for running rundaily.sh through cron
* rundaily.sh         template for running PyCIS as a live daily algorithm
* setup.sh -          install dependencies, including PyCIS-LSD

* PYLIB:
    * detect_outliers -      detect 2nd-order meaningful lines, given many spurious detections 
    * detect_outliers_agg -  detect 2nd-order meaningful lines using hierarchical clustering
    * import_fits -          import fits time-series data into data cubes, and update fits headers
    * main pipe -            control the main pipleine  
    * print_detections -     construct video/image output with detection plots
    * run_astrometry -       perform plate solving using astrometry.net and construct new headers 
    * run_pycis_lsd -        control pipeline relating to the 1st-order line segment detection from dense data cubes

* PYLSD: 
    * Built by installing PyCIS-LSD through setup.sh 

**Output Files**:
Input and output locations and names are specified within the main file, see `<runpycis.py>` for example.

* Input: data/...
    * yyymmdd_norad_satname/*.fit - a folder with raw fits frames [see citations in DATASET section below].  This input is piped to the PyCIS-LSD module.
* Output: 
    * results/...
        * record_name.json - potential ASO ra/dec measurements of the line-of-sight relative to the sensor - positioning information should be provided in FITS headers.  [TO BE ADDED: name.kvn - format as tracking data message (TDM) according to CCSDS standard]. 
        * img_name.png - still 2D image with projected detections (potential ASOs printed in red).  The background may be suppressed for viewing. 
        * VideoAll_name.gif - animation of detections, rejected lines colored by time of appearance and potential ASOs colored red.
        * VideoObj_name.gif - animation with only the potential ASOs plotted.
    * results_work/... 
        * data1_x_y_name.npy - edge line detections, x and y labels for parallelization are absent when merged.  This output is produced by the PyCIS-LSD module.
        * data2_x_y_name.npy - center line detections, x and y labels for parallelization are absent when merged.  This output is produced by the PyCIS-LSD module.
        * goodlines_name.npy - 2nd-order meaningful detections (potential ASOs)
        * badlines_name.npy - 1st-order meaningful only detections (stars, sensor structures, etc)
        * CLUST{1,2,3,X}, OUTLIER, and REMAIN_name.npy - partitionings of the 2nd-order clustering process, for some plotting functions.  
    * ../newdata/yyymmdd_norad_satname/*.fit - updated fits frames with detection results in headers, with new header keys: 
        * NUM_PY - number of detections, differentiated as "#" from 0 to NUM_PY in following 
        * RA_PY_# - right ascension in FK5 refernce frame (hours)
        * DEC_PY_# - declination in FK5 refernce frame (degrees)
        * NFA_PY_# - NFA value of detected centerline 

------------------------------------------------------------------

## REFERENCES 
(see REFERENCES.txt for full citations)

* ASTRIANet data resources: [Feuge-Miller_2021]
* Background NFA formulations: [Desolneux_2000] [Desolneux_2008] [Desolneux_2016]
* Initial LSD algorithm: See the PyCIS-LSD repo
 
------------------------------------------------------------------

## NOTICE:
 
PyCIS: An a-contrario detection algorithm for space object tracking from optical time-series telescope data. 
Copyright (C) 2022, Benjamin G. Feuge-Miller, <benjamin.g.miller@utexas.edu>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

------------------------------------------------------------------
