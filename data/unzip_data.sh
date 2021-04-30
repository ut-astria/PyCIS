#!/bin/bash

backhome=$(pwd)
for folder in *; do
	cd $folder
	for zip in *.zip; do
		unzip $zip
		rm $zip
	done
	cd $backhome
done