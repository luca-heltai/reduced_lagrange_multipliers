#!/bin/bash

outputDir="./output/sensitivity_inclusions_2"

h3d1dList=(0.1 0.01 0.001 0.5 0.05 0.005 0.0001 0.0005)
nrefList=(5 10 20 50 100)
for h3d1d in "${h3d1dList[@]}"; do
    for inc_ref in "${nrefList[@]}"; do

        outputFolder=${outputDir}/h${h3d1d}ref${inc_ref}
        #remove outpur dir if it exists already
        if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
        # create output directory
        mkdir ${outputFolder}

        path_to_input_file_3d="input_file/input_3d.prm"

        # modify the output directory and dt in the input files
        sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'|' $path_to_input_file_3d
        sed -i 's|\(set 3D 1D discretization\).*|set 3D 1D discretization                = '${h3d1d}'|' $path_to_input_file_3d
        sed -i 's|\(set Data file\).*|set Data file                           = data/data_sensitivity/inclusions_data_bifurcation_'${h3d1d}'.txt|' $path_to_input_file_3d
        sed -i 's|\(set Inclusions file\).*|set Inclusions file                     = data/data_sensitivity/inclusions_points_bifurcation_'${h3d1d}'.txt|' $path_to_input_file_3d
        sed -i 's|\(set Inclusions refinement\).*|set Inclusions refinement               = '${inc_ref}'|' $path_to_input_file_3d

        #execute command
        mpirun -np 12 ./build/elasticity_debug $path_to_input_file_3d >& ${outputFolder}/log
        process_id=$!
        wait $process_id
        echo "completed " $outputFolder ${process_id}
    done
done


## ./execute >& log

# cd ${outputDir}
# for d in */ ; do
#     cd $d
#     echo $d
#     h5dump externalPressure.h5
#     cd ..
# done
