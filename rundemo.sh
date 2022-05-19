#!/bin/bash
#
#Run demonstration scripts
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
#--------------------------------------------------------------------

CWD=$(pwd)
COPYLOC="$CWD/data"
OUTLOC="$CWD/results"
TDMLOC="$CWD/TDM"
#Create folders if necessary
if [ ! -d "$COPYLOC" ]; then
    echo "ERROR: PLEASE DOWNLOAD DEMO DATA INTO THE DATA FOLDER"
    echo "aborting..."
    return 0
fi
if [ ! -d "$OUTLOC" ]; then
    mkdir $OUTLOC
fi
if [ ! -d "$TDMLOC" ]; then
    mkdir $TDMLOC
fi

timer=60 #runtime in minutes
sleeper=2 #sleep time in minutes (to close scripts)
for FOLDER in $COPYLOC/*/; do
FOLDER=${FOLDER%*/}
FOLDER="${FOLDER##*/}"




if ! type "sbatch" >& /dev/null; then
#IF sbatch does not exist (standard case)
echo "launching on bash"
#reset paths
if ! which "solve-field" >& /dev/null ; then
echo "resetting paths"
. setup.sh
fi

# Check installation and activate source 
source env/bin/activate
#set omp and run demo with faulthandler
export OMP_NUM_THREADS=48
python3 -q -X faulthandler demo.py
python3 -q -X faulthandler runpycis.py -i $COPYLOC -s $FOLDER -o $OUTLOC/$FOLDER -t $TDMLOC
#close source for safe exit
deactivate

else
#IF sbatch does exist (TACC case)
echo "launching through slurm"
#reset paths
module load gcc/9.1
module load cmake/3.16.1 #.10.2
module load python3/3.8.2
if ! which "solve-field" >& /dev/null ; then
echo "resetting paths"
. setup.sh
fi
#Launch sbatch 
cat <<EOF |sbatch
#!/bin/bash
#SBATCH -p skx-normal
#SBATCH -J pycis.job
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:$timer:00
#SBATCH -o $COPYLOC/$FOLDER/pycis.out

# #SBATCH --mail-user 
# #SBATCH --mail-type ALL

# Switch to folder and load modules
module load gcc/9.1
module load cmake/3.10.2
module load python3/3.8.2

# Check installation and activate source 
source env/bin/activate

#set omp and run demo with faulthandler
export OMP_NUM_THREADS=48
python3 -q -X faulthandler runpycis.py -i $COPYLOC -s $FOLDER -o $OUTLOC/$FOLDER -t $TDMLOC

#close source for safe exit
deactivate
EOF
fi

done
