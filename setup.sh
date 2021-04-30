#! bin/bash

CWD=$(pwd)

# install gsl
if [ ! -d "./gsl" ]; then 
    export CC=`which gcc`
    export CFLAGS="-O3 -march=native"
    # get gsl
    wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.6.tar.gz
    tar -zxvf gsl-2.6.tar.gz
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
    rm gsl-2.6.tar.gz
fi
export LD_LIBRARY_PATH=./gsl/lib:$LD_LIBRARY_PATH

# install venv
if [ ! -d "./env" ]; then
    python3 -m venv ./env
    source env/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    deactivate
fi

#activate env
source env/bin/activate

#install library sources
cd lib
make
cd ${CWD}

#python3 setup.py install --user
python3 setup.py build_ext --inplace

deactivate

