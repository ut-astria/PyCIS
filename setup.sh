#!/bin/bash
#
#Assumes python 3.8, cmake, and gcc are installed
#Download, install, clean up, and set paths for:
#   0) PyCIS-LSD software
#   1) Python3 environment 
#   2) Point alignment software
#       2a) FLANN
#       2b) 3d_point_alignments
#   3) Astrometry software
#       3a) CFITSIO
#       3b) WCSlib
#       3c) Astrometry.net offline software
#Install astrometry index files for solving
#Then make the C library and launch python setup to link PyCIS
#
#--------------------------------------------------------------------
#PyCIS: An a-contrario detection algorithm for space object tracking from optical time-series telescope data. 
#Copyright (C) 2022, Benjamin G. Feuge-Miller, <benjamin.g.miller@utexas.edu>
#
#This program is free software; you can redistribute it and/or modify
#t under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 2 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License along
#with this program; if not, write to the Free Software Foundation, Inc.,
#51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#--------------------------------------------------------------------
##LOAD INITIAL DEPENDENCIES 
if type "sbatch" >& /dev/null; then
    module load gcc/9.1
    module load cmake/3.16.1 #10.2
    module load hdf5
    module load python3/3.8.2
fi
export CC=`which gcc`
export CFLAGS="-O3 -march=native"
#Save directory for path resetting
CWD=$(pwd)

#--------------------------------------------------------------------
## INSTALL PYTHON ENVIRONMENT
#some systems may need python3.6 specifically if 3.8+ is not available 
if [ ! -d "./env" ]; then
    python3 -m venv ./env
fi
if [ ! -d "./env/lib/python3.8/site-packages/imageio" ]; then
    source env/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    deactivate
fi
#Install the PyCIS-LSD library 
if [ ! -d "./pylsd" ]; then
    git clone https://github.com/ut-astria/PyCIS-LSD.git "./pylsd"
    dos2unix pylsd/*.sh
fi
export PATH=$PATH:$CWD/pylsd
cd pylsd
. setuplsd.sh
cd ${CWD}

#--------------------------------------------------------------------
## INSTALL SOFTWARE FOR TRACK ASSOCIATION 
#Flann library 
if [ ! -d "./flann" ]; then
    git clone https://github.com/mariusmuja/flann.git
    cd flann
    #Necessary for cmake>=3.11
    touch src/cpp/empty.cpp
    sed -i '33s#.*#    add_library(flann_cpp SHARED empty.cpp)#' src/cpp/CMakeLists.txt
    sed -i '91s#.*#        add_library(flann SHARED empty.cpp)#' src/cpp/CMakeLists.txt
    #Continue build
    mkdir build
    cd build
    cmake .. -DBUILD_C_BINDINGS=ON -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_DOC=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF
    make
    cd ${CWD}
fi
export PATH=$PATH:${CWD}/flann/src/cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CWD}/flann/build/lib
#"An Unsupervised Point Alignment Detection Algorithm"
#by J. Lezama, G. Randall, J.M. Morel and R. Grompone von Gioi,
#Image Processing On Line, 2015. http://dx.doi.org/10.5201/ipol.2015.126
if [ ! -d "./ptalign" ]; then
    wget -c http://www.ipol.im/pub/art/2017/214/3d_point_alignments_1.0.zip
    unzip 3d_point_alignments_1.0.zip -d ptalign
    rm -r 3d_point_alignments_1.0.zip
    cd ptalign
    sed -i '9s#.*#LIB_DIRS=../flann/build/lib#' Makefile
    sed -i '10s#.*#INC_DIRS=../flann/src/cpp#' Makefile
    make omp WITH_FLANN=true
    cd ${CWD}
fi
export PATH=$PATH:${CWD}/ptalign
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CWD}/ptalign

#--------------------------------------------------------------------
## INSTALL SOFTWARE FOR ASTROMETRY
# install cfitsio
if [ ! -d "./cfitsio" ]; then 
    # get astrometry
    wget -c http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz
    tar -zxf cfitsio_latest.tar.gz
    rm cfitsio_latest.tar.gz 
    # prepare paths
    mkdir cfitsio
    cd cfitsio-*
    # install 
    ./configure --enable-sse2 --prefix=${CWD}/cfitsio --enable-reentrant
    make
    make install 
    #make testprog
    #./testprog > testprog.lis
    #diff testprog.lis testprog.out
    #cmp testprog.fit testprog.std
    # clean up
    cd ${CWD}
    rm -rf cfitsio-*
fi
export PATH=$PATH:$CWD/cfitsio/bin
export PATH=$PATH:$CWD/cfitsio/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CWD/cfitsio/lib
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:$CWD/cfitsio/lib/pkgconfig/
# install wcslib
if [ ! -d "./wcslib" ]; then 
    # get astrometry
    wget -c ftp://ftp.atnf.csiro.au/pub/software/wcslib/wcslib-7.6.tar.bz2
    tar -jxf wcslib-7.6.tar.bz2
    rm wcslib-7.6.tar.bz2
    # prepare paths
    mkdir wcslib
    cd wcslib-*
    # install 
    ./configure --without-pgplot --disable-fortran --prefix=${CWD}/wcslib
    make
    make install 
    # clean up
    cd ${CWD}
    rm -rf wcslib-*
fi
export PATH=$PATH:$CWD/wcslib/bin
export PATH=$PATH:$CWD/wcslib/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CWD/wcslib/lib
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:$CWD/wcslib/lib/pkgconfig/
#Install Astrometry.net offline core software (without extra packages for printing)
#source env/bin/activate
if [ ! -d "./astrometry" ]; then 
    # get astrometry
    wget -c http://astrometry.net/downloads/astrometry.net-latest.tar.gz
    tar -zxf astrometry.net-latest.tar.gz
    rm astrometry.net-latest.tar.gz #req for pathing
    # prepare paths
    mkdir astrometry
    cd astrometry.net*
    # install astrometry without plotting functions (make extra)
    # use gsl installed above
    ./configure --prefix=${CWD}/astrometry
    make reconfig
    make SYSTEM_GSL=yes GSL_inc="-I${CWD}/gsl/include" GSL_LIB="-L${CWD}/gsl/lib -lgsl" CFITS_INC="-I${CWD}/cfitsio/include" CFITS_LIB="-L${CWD}/cfitsio/lib -lcfitsio" WCSLIB_INC="-I${CWD}/wcslib/include" WCSLIB_LIB="-L${CWD}/wcslib/lib -lwcslib"
    make install INSTALL_DIR=${CWD}/astrometry
    # clean up
    cd ${CWD}
    rm -rf astrometry.net-*
    #We use 1.76 deg (105.6 arcmin) FOV for ASTRIANet NMSkies
    #Astrometry.net recommends 10%-100% diameter index files (11-110 arcmin)
    #get wide-angle index files (30-2000 arcmin diameter skymarks), easy to store
    for ss in {4208..4219}; do
        wget -c -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/4100/index-${ss}.fits" #Tycho-2
        wget -c -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/4200/index-${ss}.fits" #2MASS
    done
    #Get medium-sized index files (11-30 arcmin diameter skymarks)
    for ss in {4205..4207}; do
        for ssa in {00..11}; do
            ssb=$(printf "%02d" ${ssa})
            wget -c -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/4200/index-${ss}-${ssb}.fits" 
        done
    done

    #for ss in {5000..5007}; do
    #    for ssa in {00..11}; do
    #        ssb=$(printf "%02d" ${ssa})
    #        wget -c -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/5000/index-${ss}-${ssb}.fits" 
    #    done
    #done

    #CONTINGENCY: Prevent duplicate download
    rm "${CWD}/astrometry/data/*.fits.*"
    #Get small-sized index files (2-11 arcmin diameter skymarks)
    #for ss in {4200..4204}; do
    #    for ssa in {00..47}; do
    #        ssb=$(printf "%02d" ${ssa})
    #        wget -c -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/4200/index-${ss}-${ssb}.fits" 
    #done   
fi
#IF THERE IS A BAD INSTALL IN usr/local, CAN INSTEAD 
#SWTICH PRIORITY BY PATH=/dir:$PATH
export PATH=${CWD}/astrometry/bin:$PATH
export PATH=$CWD/astrometry/include:$PATH
export LD_LIBRARY_PATH=$CWD/astrometry/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$CWD/astrometry/lib/pkgconfig/:${PKG_CONFIG_PATH}

#deactivate

