#!/bin/bash
#Get ASTRIANet data from:
#Feuge-Miller, Benjamin; Kucharski, Daniel; Iyer, Shiva; Jah, Moriba, 2021, 
#"ASTRIANet Data for: Python Computational Inference from Structure (PyCIS)",
#https://doi.org/10.18738/T8/GV0ASD, Texas Data Repository, V3  

if [ ! -d "data" ]; then
    mkdir "data"
fi
CWD=$(pwd)
#FOLDER="20201224_26407_navstar-48"
#LOC="data/20201220_45696_starlink-1422"
#if [ ! -d "${LOC}" ]; then
#    mkdir "${LOC}"
#    cd "${LOC}"
#    wget -c https://dataverse.tdl.org/api/access/datafile/123198 -O temp.tar.gz
#    tar -zxf temp.tar.gz --strip-components=1
#    rm -rf temp.tar.gz
#   cd $CWD
#fi
LOC="data/20201224_26407_navstar-48"
if [ ! -d "${LOC}" ]; then
    mkdir "${LOC}"
    cd "${LOC}"
    wget -c https://dataverse.tdl.org/api/access/datafile/123197 -O temp.tar.gz
    tar -zxf temp.tar.gz --strip-components=1
    rm -rf temp.tar.gz
    cd $CWD
fi
