#!/bin/bash
#A simple cron-job to launch rundaily.sh regularly 
NOW=$(date +"%Y%m%d")
. rundaily.sh &> $NOW-cron.log

