#! bin/bash

# We first install GSL, required for the LSD algorithms and Astrometry.net
# Next the python environment is installed to build Astropy
# Then we install (in order) cfitsio, pgplot, and wcslib
# Then install astrometry.net with specific index files for plate solving
# We conclude by activating the environment and building PyCIS itself

CWD=$(pwd)

# install gsl
if [ ! -d "./gsl" ]; then 
    export CC=`which gcc`
    export CFLAGS="-O3 -march=native"
    # get gsl
    wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.6.tar.gz 
    tar -zxf gsl-2.6.tar.gz
    rm gsl-2.6.tar.gz
    # prepare paths
    mkdir gsl
    cd gsl-2.6
    # install gsl
    ./configure --prefix=${CWD}/gsl
    make -j 12
    make check
    make install
    # clean up
    cd ${CWD}
    rm -r gsl-2.6
    
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./gsl/lib

# install venv using gsl, to install astropy
if [ ! -d "./env" ]; then
    python3.7 -m venv ./env
    source env/bin/activate
    python3.7 -m pip install --upgrade pip
    python3.7 -m pip install -r requirements.txt
    deactivate
fi

# install cfitsio
if [ ! -d "./cfitsio" ]; then 
    export CC=`which gcc`
    export CFLAGS="-O3 -march=native"
    # get astrometry
    wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz
    tar -zxf cfitsio_latest.tar.gz
    rm cfitsio_latest.tar.gz 
    # prepare paths
    mkdir cfitsio
    cd cfitsio-*
    # install 
    ./configure --enable-sse2 --prefix=${CWD}/cfitsio --enable-reentrant
    make
    make install 
    # clean up
    cd ${CWD}
    rm -r cfitsio-*
fi
export PATH=$PATH:./cfitsio/bin

# install wcslib
if [ ! -d "./wcslib" ]; then 
    export CC=`which gcc`
    export CFLAGS="-O3 -march=native"
    # get astrometry
    wget ftp://ftp.atnf.csiro.au/pub/software/wcslib/wcslib-7.6.tar.bz2
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
    rm -r wcslib-*
fi
export PATH=$PATH:./wcslib/bin

# install astrometry.net core (without extra packages for printing)
if [ ! -d "./astrometry" ]; then 
    export CC=`which gcc`
    export CFLAGS="-O3 -march=native"
    # get astrometry
    wget http://astrometry.net/downloads/astrometry.net-latest.tar.gz
    tar -zxf astrometry.net-latest.tar.gz
    rm astrometry.net-latest.tar.gz #req for pathing
    # prepare paths
    mkdir astrometry
    cd astrometry.net*
    # install astrometry without plotting functions (make extra)
    # use gsl installed above
    ./configure --prefix=${CWD}/astrometry
    make SYSTEM_GSL=yes GSL_inc="-I./gsl/include" GSL_LIB="-L./gsl/lib -lgsl" \
        CFITS_INC="-I./cfitsio/include" CFITS_LIB="-L./cfitsio/lib -lcfitsio" \
        WCSLIB_INC="-I./wcslib/include" WCSLIB_LIB="-L./wcslib/lib -lwcslib"
    make install INSTALL_DIR=${CWD}/astrometry
    #make reconfig
    # clean up
    cd ${CWD}
    rm -r astrometry.net-*
    #We use 1.76 deg (105.6 arcmin) FOV for ASTRIANet NMSkies
    #Astrometry.net recommends 10%-100% diameter index files (11-110 arcmin)
    #get wide-angle index files (30-2000 arcmin diameter skymarks), easy to store
    for ss in {4208..4219}; do
        wget -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/4100/index-${ss}.fits" #Tycho-2
        wget -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/4200/index-${ss}.fits" #2MASS
    done
    #Get medium-sized index files (11-30 arcmin diameter skymarks)
    for ss in {4205..4207}; do
        for ssb in {00..11}; do
            ssb=$(printf "%02d" ${ssa})
            wget -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/4200/index-${ss}-${ssb}.fits" 
        done
    done
    #Get small-sized index files (2-11 arcmin diameter skymarks)
    #for ss in {4200..4204}; do
    #    for ssa in {00..47}; do
    #        ssb=$(printf "%02d" ${ssa})
    #        wget -r -nd -np -P ${CWD}/astrometry/data "data.astrometry.net/4200/index-${ss}-${ssb}.fits" 
    #done  
fi
export PATH=$PATH:./astrometry/bin

#activate env
source env/bin/activate

#install library sources
cd lib
make
cd ${CWD}

#python3 setup.py install --user
python3 setup.py build_ext --inplace

deactivate

