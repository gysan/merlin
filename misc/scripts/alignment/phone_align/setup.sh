#!/bin/bash

if test "$#" -ne 0; then
    echo "Usage: ./setup.sh"
    exit 1
fi

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $(dirname $current_working_dir))))
### default settings ###
config_file=config.cfg

echo "######################################" > $config_file
echo "############# PATHS ##################" >> $config_file
echo "######################################" >> $config_file
echo "" >> $config_file

echo "MerlinDir=${merlin_dir}" >>  $config_file
echo "frontend=${merlin_dir}/misc/scripts/frontend" >> $config_file
echo "WorkDir=${current_working_dir}" >>  $config_file
echo "" >> $config_file

echo "######################################" >> $config_file
echo "############# TOOLS ##################" >> $config_file
echo "######################################" >> $config_file
echo "" >> $config_file

echo "ESTDIR=${merlin_dir}/tools/speech_tools" >> $config_file
echo "FESTDIR=${merlin_dir}/tools/festival" >> $config_file
echo "FESTVOXDIR=${merlin_dir}/tools/festvox" >> $config_file
echo "" >> $config_file

echo "Path to Merlin and other tools configured in $config_file"
echo "setup done...!"
