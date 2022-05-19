#!/bin/bash
#
#A script template for regularly fetching data as called by runcron.sh, 
#launching the runpycis.py processing, and pushing output data if available
#
#TODO: 
#new get-data script to read any new folder not in some record.log file 
#
#Benjamin Feuge-Miller: benjamin.g.miller@utexas.edu
#The University of Texas at Austin, 
#Oden Institute Computational Astronautical Sciences and Technologies (CAST) group
#Date of Modification: May 5, 2022
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




#Set environment variables 
#cd $SCRATCH/TIP

#Get list of new folders
CWD=$(pwd)

ASTRIA_MEASUREMENTS= #Insert location to data folders containing NMSkies telescopes folders
if [ -z ${ASTRIA_MEASUREMENTS} ]; then
    echo "ERROR:  PLEASE SPECIFY DATA FOLDER ASTRIA_MEASUREMENTS IN RUNDAILY.SH.  EXITING..."
    return 0
fi

#NMSkies data and Emerald data
declare -a ASTRIA_LIST=("NMSkies_FLI0" "EmeraldAU_FLI0")

#Local copy of all ASTRIANET data
COPYLOC="${SCRATCH}/TIP2data"
#Local copy of results, for temporary wriring
OUTLOC="${SCRATCH}/TIP2results"
#Ultimate location where to push TDMs for ASTRIAGraph
#TDMLOC="{$ASTRIA_MEASUREMENTS}/ImagePipeline/"
TDMLOC="TDM"

#Create folders if necessary 
if [ ! -d "$COPYLOC" ]; then
    mkdir $COPYLOC
fi
if [ ! -d "$OUTLOC" ]; then
    mkdir $OUTLOC
fi
if [ ! -d "$TDMLOC" ]; then
    mkdir $TDMLOC
fi

#Get any new data from corral, no rsync needed
#NOTE: ASSUMES UNZIPPED
for ASTRIA_FOLDER in "${ASTRIA_LIST[@]}"; do
    cp -R -u -p "${ASTRIA_MEASUREMENTS}/${ASTRIA_FOLDER} ${COPYLOC} "  
done

#Legacy: Count fit files in local diretory 
#echo "Counting fit and fits files..."
#typeset -i fitcount=$(ls -l $COPYLOC/*/*.fit | wc -l)
#typeset -i fitscount=$(ls -l $COPYLOC/*/*.fits | wc -l)
#filecount=$(($fitcount+$fitscount))

#Set up PYCIS
module load gcc/9.1
module load cmake/3.16.1 #.10.2
module load python3/3.8.2
module load hdf5
#. setup.sh
if ! which "solve-field" >& /dev/null ; then
    echo "resetting paths"
    . setup.sh
fi

#Launch sbatch 
timer=120 #runtime in minutes
WINDOW=30
#sleeper=2 #sleep time in minutes (to close scripts)
for FOLDER in $COPYLOC/*/; do
FOLDER=${FOLDER%*/}
FOLDER="${FOLDER##*/}"
#NOTE: using these calls, we're requesting separate jobs for unique result folder locations, 
#so that there is no cross-contamination.  
echo "Writing output to ${COPYLOC}/${FOLDER}/pycis.out"
cat <<EOF |sbatch
#!/bin/bash
#SBATCH -p skx-normal
#SBATCH -J pycis_${FOLDER}_${WINDOW}.job
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:$timer:00
#SBATCH -o $COPYLOC/$FOLDER/pycis_${WINDOW}_${FOLDER}.out

# #SBATCH --mail-user 
# #SBATCH --mail-type ALL

# Switch to folder and load modules
module load gcc/9.1
module load cmake/3.16.1
module load python3/3.8.2
module load hdf5

# Check installation and activate source 
source env/bin/activate

#set omp and run demo with faulthandler
export OMP_NUM_THREADS=48
python3 -q -X faulthandler runpycis.py -i ${COPYLOC} -s ${FOLDER} -o ${OUTLOC}/${FOLDER} -t ${TDMLOC} -w $WINDOW

#close source for safe exit
deactivate
EOF
#parallel jobs, 'give exit time'
#sleep "$sleeper"m
#TO put to sleep longer (one job at a time) call: 
#sleep "$(($sleeper+$timer))"m
#done

#TRANSFER DATA
#echo "Transferring meaurements..."
#cp $OUTLOC/*/*.kvn $TDMLOC
#echo "Complete"


done

