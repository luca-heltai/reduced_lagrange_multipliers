#!/bin/bash

outputDir="./output/bifurcation"
dt_list=(0.5 0.1)
input1D=prova_1d
input3D=prova_3d

for dt in "${dt_list[@]}"; do
    outputFolder=${outputDir}_${dt}

    #remove outpur dir if it exists already
    if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi

    # create output directory
    mkdir ${outputFolder}
    mkdir ${outputFolder}/3D

    # create the input files as copy of the originals directly in the output folder
    path_to_input_file_1d="input_file/${input1D}_${dt}.dat"
    path_to_input_file_3d="input_file/${input3D}_${dt}.prm"

    #remove input files if they exist already
    if [ -f "$path_to_input_file_1d" ] ; then rm "$path_to_input_file_1d"; fi
    if [ -f "$path_to_input_file_3d" ] ; then rm "$path_to_input_file_3d"; fi

    cp input_file/prova_1d.dat $path_to_input_file_1d
    cp input_file/prova_3d.prm $path_to_input_file_3d

    # modify the output directory and dt in the new input files
    sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
    sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d
    sed -i 's|\(dtMaxLTSLIMIT\).*|dtMaxLTSLIMIT ='${dt}'|' $path_to_input_file_1d

    sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d

    #execute command
    ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 10 >& ${outputFolder}/log
done
