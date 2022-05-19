# PyCIS : Python Computational Inference from Structure


    An a-contrario detection algorithm for space object tracking from optical time-series telescope data. 


Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
The University of Texas at Austin, 
Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
*Date of Modification: May 19, 2022*

**NOTICE**: For copyright and licensing, see bottom of readme

------------------------------------------------------------------

## OVERVIEW:


This software takes in a time-series of optical FITS data and produces a CCSDS-formatted Tracking Data Message. This algorithm was initially presented in:
* B. Feuge-Miller, M. Jah, “A-Contrario Structural Inference for Space Object Detection and Tracking”, *in 8th Annual IAA Space Traffic Management Conference*, Austin, TX, 2022


This software aims to enhance traditional techniques for space object detection by mitigating processing requirements such as background subtraction, directly leveraging temporal information, and avoiding contrast-depentent measurements. The method does not require training data, and may be used to propagate uncertainty from data aquisition to the orbit determination processes.  This software can be used on either sidereal or rate-tracked data as potential ASOs have anomalous behavior relative to the starfield and background noise in either context.  This detection method is not designed for photometric analysis.


Under the *a-contrario* paradigm, structures in data are considered "meaningful" if they are unlikely to occur by chance.  In contrast to standard p-significance hypothesis testing, the *a-contrario* approach controls the number of detections in expectation within the data rather than by controling a probability of detection. PyCIS uses a recursive sequence of *a-contrario* steps to detect potential ASOs without making any high-fidelity *a-priori* model of predicted objects or the complicated noise structures (atmospheric, hardware, or stellar noise).  The algorithm first finds trajectories of objects surprising in the general atmospheric/sensor noise, and then infers structures of star and hardware behaviors.  The final detection is the set of features which cannot be attributed to any other noise infered from the data.  


Astrometric positioning from the detected class of star features is provided by the offline Astrometry.Net software.  This enables output of time-tagged right asension/ declination Tracking Data Messeges of the line-of-sight vector from raw optical data for further processing. In post-processing, we use localized source extraction around the results with inferred kernel matching to improve non-linear tracking capability and assist in false posititive rejection.  

------------------------------------------------------------------

## DEMO VISUAL OBSERVATION


This example can be recovered by running `<. getdata.sh>` and `<. rundemo.sh>`. The "potential ASO" detections are marked in red from post-processing, given the *a-contrario* trajectory detections shown in green. The Navstar-48 (Medium Earth Orbit) GPS satellite is used as an example to demonstrate effectiveness on low-tracking-noise active tracking (MEO-GEO objects) and object discovery.  The actively tracked Navstar is in the center of the frame and an Unexpected Orbital Object is in the lower right. Tracking Data Messages can be found in the `<docs>` folder for both detections. The data and results can be interactively visualized at [Interactive Starlink Plot](https://rawcdn.githack.com/ut-astria/PyCIS/d1c7a943fb445505a5d87da48b07fb0446b5a90a/docs/a0t10m1b2_ASSOC1T60W20O0_e0_inj-1_20201220_45696_starlink-1422_FINALVOL.html) for further visualization. 

![Demo Video](docs/20201224_26407_navstar-48.gif)

Demo script takes approximatly 3hrs (with 1-pixel binning and windows of 20 and 30 frames), for a 93-frame 11-minute observation.  The implementation is not yet optimized, and future speed-up is expected.  This was tested on a single Skylake node of the Stampede2 system of the Texas Advanced Computing Center at the University of Texas at Austin.  

------------------------------------------------------------------

## DATASET

The demo datasets from the New Mexico Skies telecope of the ASTRIANet telescope network are available at the Texas Data Repository: 

Feuge-Miller, Benjamin; Kucharski, Daniel; Iyer, Shiva; Jah, Moriba, 2021, 
"ASTRIANet Data for: Python Computational Inference from Structure (PyCIS)", 
https://doi.org/10.18738/T8/GV0ASD, Texas Data Repository, V3  

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

* ASTRIANet
    * B. Feuge-Miller, M. Jah, “A-Contrario Structural Inference for Space Object Detection and Tracking”, *in 8th Annual IAA Space Traffic Management Conference*, Austin, TX, 2022
    * Feuge-Miller, Benjamin; Kucharski, Daniel; Iyer, Shiva; Jah, Moriba, 2021, "ASTRIANet Data for: Python Computational Inference from tructure (PyCIS)", https://doi.org/10.18738/T8/GV0ASD, Texas Data Repository, V3  
* A-Contrario Theory
    * A. Desolneux, L. Moisan, J.-M. Morel, Meaningful alignments, International journal of computer vision 40 (1) (2000) 7–23. doi:10.1023/A:1026593302236.
    * A. Desolneux, L. Moisan, J.-M. Morel, From gestalt theory to image analysis: a probabilistic approach, Vol. 34, Springer Science & Business Media, 2007. doi:https://doi.org/10.1007/978-0-387-74378-3.

 
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
