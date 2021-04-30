# PyCIS - Python Computational Inference from Structure

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
*Date of Modification: April 30, 2021*

**NOTICE**: For copyright and licensing, see 'notices' at bottom of readme

------------------------------------------------------------------

## OVERVIEW:

Under the so-called "a-contrario" paradigm, structures in data are considered "meaningful"
if they are unlikely to occur by chance according to a background noise model.  
This "Helmholtz Principle" is formalized by a "Number of False Alarms" (or NFA) function, 
probabilistically computing the likelihood of a structure's existence given 
both a noise model assumption and some measurement function.
We are interested in detecting moving features in noisy video data, organized as a 
three-dimensional data cube, under which the object is a linear feature.  
Building on existing line-segment-detection algorithms for dense 2D images, 
we use the NFA framework to measure edge surface features in the video data
given no prior information, perform a second round of NFA analysis to detect 
center line features using the new prior edge information, and finally perform 
second-order a-contrario analysis for outlier detection.
This second-order analysis accounts for spurious structures in noise
(e.g. star motion, telescope noise) which have a large number of detectable features 
and hence should be rejected under the paradigm.  
See 'Operation' and 'Notes on Demo Output Visual Observations' sections below.


------------------------------------------------------------------

## OPERATION:

Bash scripts are provided which will 
1) launch prerequisite software installation of GSL 
2) create and/or activate python3 environment "./env"
3) compile the C library given in "./lib"
4) link the C library with gsl as a python extension module
5) run the demo script "demo.py"

To launch the scripts, run the following appropriate command:

Setup:
If on TACC, load the following: 
"module load gcc/9.1; module load python3/3.8.2"
And launch the installation ("." bash call allows correct pathing):
". setup.sh"

Demo:
For TACC systems:
"sbatch pycis_demo.job"
For all other systems:
". run_demo.sh"
and ensure 'framerange' and 'scale' are reduced on demo.py, to prevent memory crash.

The demo will:
1) read a subset of fits files from data/ (using pylib/import_fits). 
2) performs line segment detection (LSD) through pycis.c 
    2a) detect parallel and orthogonal markov kernels 
    2b) perform first-order gestalt detection of edge-lines
    2c) perform first-order gestalt detection of center-lines with edge prior
3) perform second-order gestalt detection (using pylib/detect_outliers)
4) print results to results/. 

The second-order-meaningful lines correspond to detection of a 
Starlink satellite made using ASTRIANet telescope resources 
through UT CAST [see citations in DATASET section below]. 

Input: data/...
    yyymmdd_norad_satname/*.fit - a folder with raw fits frames
        [see citations in DATASET section below] 
Output: results/...
    data1_name.npy - edge line detections
    data2_name.npy - center line detections
    goodlines_name.npy - 2nd-order meaningful detections
    badlines_name.npy - 1st-order meaningful only detections
    img_name.npy - last fit frame data with projected detections (2nd order in red)
    vidAll_name.npy - animated fits data with projected detections (2nd order in red)
    vidObj_name.npy - animated fits data with only 2nd-order projected detections

Runtime estimates are ~30 min on TACC for an unscaled 26-frame subset (4906x4906px)
    ~10min for 0.5 scaling, but GMM model fitting requires performance 
    improvements on reduced point sets to accuratly seperate outliers. 
Runtime not yet estimated for local machines due to size of full-scale data sets.  

------------------------------------------------------------------

## NOTES ON DEMO OUTPUT VISUAL OBSERVATION


Many single-frame features (namely star-streak features) 
are visual gestalts but are not registered as 'meaninful events'
(as of version 0.1).
This is likely due to how the current region-growing algorithm of the 
measurement function handles accumulation of polar angles, and 
is slated for a near-term update. 

On the frame subset of the demo, the track is lost on the last frames 
of the vidObj*.avi video.  
This is likely due to linearity constraints on region-growing of center-
line features (non-constant velocity), and is slated for a near-term update.

------------------------------------------------------------------

## DATASET

The data/20201220_45696_starlink-1422 folder contains
a sequence of 66 FITS-format data frames collected from the 
NMSkies telecope of the UTA-ASTRIA ASTRIA-Net telescope network.
Specifically, the frames are a track of the Starlink-1422 satellite 
on the night of Dec. 20, 2020.  Relevant observation data can be found 
in the FITS headers, printable by turning on the 'headerflag' variable in 
import_fits.py.  See citations below.  

Maria Esteva, Weijia Xu, Nevan Simone, Amit Gupta, Moriba Jah. 
Modeling Data Curation to Scientific Inquiry: 
A Case Study for Multimodal Data Integration. 
The ACM/IEEE Joint Conference on Digital Libraries in 2020. 
June 19-23, Wuhan, China. Doi: https://doi.org/10.1145/3383583.3398539 

Simone, Nevan; Nagpal, Kartik; Gupta, Amit; Esteva, Maria; Xu, Weijia; Jah, Moriba, 
2021, "Replication Data for: Transparency and Accountability in Space Domain Awareness: 
Demonstrating ASTRIAGraph's Capabilities with the United Nations Registry Data", 
https://doi.org/10.18738/T8/NBWWWZ, Texas Data Repository, V1 

The authors acknowledge the Texas Advanced Computing Center (TACC) 
at The University of Texas at Austin for providing 
HPC and database resources that have contributed to the research software, 
particularly in computing resources for running the software and in 
storage capacity for hosting ASTRIA-Net data. 
URL: http://www.tacc.utexas.edu

------------------------------------------------------------------

## TOC:

demo.py -           run demo as discussed above
pycis_demo.job -   slurm call to run_demo for TACC
pycis.c -          main python extension module 
run_demo.sh -       activate env and launch demo
setup.py -          link python-c extension module
setup.sh -          install gsl/pyenv/Makefile, link


LIB:
    constants -     set up some constants
    gaussians -     gaussian down-sampling for antialiasing
    gradient -      compute gradient magnitude/orientation, and alignment checks
    interface -     pipeline helpers for pycis.c
    lsd(2/3) -      main lsd pipelines
    markov -        estimate markov kernels and build transition matrices
    misc -          lists, angle functions
    NFA -           estimate markov or negative binomial approximation tail probability
    rectanges(2/3)- build rectangle objects and iterator to call nfa
    regions(2/3) -  build and improve pixel regions for estimating rectangles
    tuples -        construction of line tuples and image structures 

PYLIB:
    detect_outliers -   detect 2nd-order meaningful lines, given many spurious detections 
    import_fits -       import fits time-series data into data cubes
    print_detections -  construct video/image output with detection plots

------------------------------------------------------------------

## REFERENCES 
(see REFERENCES.txt for full citations)

Data resources:
    ASTRIA-Graph [Esteva_2020, Simone_2021]

Background NFA formulations:
    Meaningful Alignments [Desolneux_2000] [Desolneux_2008] [Desolneux_2016]

Initial LSD algorithm:
    Initial LSD code: [vonGioi_2012]
    Adaption of LSD for non-independent hypothesis and robust gradients: [Liu_2019]
    Source for non-independence hypothesis [Almazan_2017] [Myaskouvskey_2013]
    Source for robust gradient-by-ratio [Dellinger_2015]

Centerline detection modifications (rectangles(2/3).c, lsd(2,3).c)
    2nd-order gestalts and the two-run NFA [Lezama_2014] [Simon_2018]
    Improved geometric-series rectangle improvement [Lezama_2015]

Other relevant attempts towards a 3D solution: 
    Trajectory method towards 3D alg: [Dimiccoli_2016]
    Point cloud method towards 3D alg: [Gomez_2017]
    Binomial framing of outlier problem: [Moisan_2016] [Zair_2015]

Outlier detection of 2nd-order gestalt (detect_outliers.py)
    Single-outlier detection, NFA as p-values [Grosjean_2008]
    Mahanalobis GMM binomial outlier with chi2 model [Rousseau_2008]
    Mahalanobis fisher model (see detect_outliers.py) [Roizman_2020]
    Weighted z-test [Zaykin_2011]

NFA approximation (nfa.c):  
    Markov binomial approximation  [Xia_2010]

ROC curve analysis:
    P-value method towards ROC curves [Maumet_2016]
    ROC Curves [Bowyer_1999] [Dougherty_1998]
 
------------------------------------------------------------------

## NOTICES:

Notice for PyCIS: see pycis.c for more details 
    **NOTICE**: This program is modified from the source code of LSDSAR:
    "LSDSAR, a Markovian a contrario framework for line segment detection in SAR images"
    by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. 
    (Pattern Recognition, 2019).
    https://doi.org/10.1016/j.patcog.2019.107034
    *Date of Modification: April 30, 2021*

Copyright notice for LSDSAR:
    **NOTICE**: This code is modified from the source code of LSD:
    "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
    Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
    Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
    http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
    *Date of Modification: 27/06/2018*

**NOTICE**: This program is released under GNU Affero General Public License
and any conditions added under section 7 in the link:
https://www.gnu.org/licenses/agpl-3.0.en.html

Copyright (c) 2021 Benjamin Feuge-Miller <benjamin.g.miller@utexas.edu>

This program is free software: you can redistribute it and/or modify 
 it under the terms of the GNU General Public License as published 
 by the Free Software Foundation, either version 3 of the License, 
 or (at your option) any later version.

This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.
 
You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

------------------------------------------------------------------