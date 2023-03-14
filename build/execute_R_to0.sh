#!/bin/bash

start=$(date +%s)
ri="0.2"
ri2="0.04"

build_dir="/workspaces/CODICE_lagrangian/build"
output_dir="output/R_to0/"
cd $output_dir

for d in */; do ## all the test folder
    folder=$(basename "$d")
    # echo $folder
    modes=${folder##*_}  # get all  after last separator '_'
    # echo $modes
    if [[ "$folder" = *01* ]]
        then
        radius=0.1
    fi
    if [[ "$folder" = *02* ]]
        then
        radius=0.2
    fi
    if [[ "$folder" = *005* ]]
        then
        radius=0.05
    fi

    inclusions_position="0.3,0.3,"$radius";-0.4,0.3,"$radius
    output_directory="output\/R_to0\/"$folder
    cd ../../
    echo $PWD
    sed -i "s/\(set Output directory\).*/set Output directory                   = "$output_directory"/g" input.prm
    sed -i "s/\(set Number of fourier coefficients\).*/set Number of fourier coefficients      = "$modes"/g" input.prm
    sed -i "s/\(set Inclusions   \).*/set Inclusions                          = "$inclusions_position"/g" input.prm
    
    ./elasticity_debug input.prm >& $output_dir$folder/log &
    process_id=$!
    wait $process_id
    echo "completed " $folder "$?"
    cd $output_dir
done
