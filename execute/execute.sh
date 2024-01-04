#!/bin/bash

outputDir="./output/bifurcation_221123"

# ###################################################################################################################################################
# # UNCOUPLED ELASTIC
# ###################################################################################################################################################
# outputFolder=${outputDir}/uncoupled_elastic
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Elastic_1d.dat"
# path_to_input_file_3d="input_file/input_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 11 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

# ###################################################################################################################################################
# # UNCOUPLED VISCO
# ###################################################################################################################################################
# outputFolder=${outputDir}/uncoupled_visco
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Visco_1d.dat"
# path_to_input_file_3d="input_file/input_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 11 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

# ###################################################################################################################################################
# # CONTROL VISCOUS
# ###################################################################################################################################################
# outputFolder=${outputDir}/control_viscous
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Visco_1d.dat"
# path_to_input_file_3d="input_file/input_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

# ###################################################################################################################################################
# # CONTROL ELASTIC
# ###################################################################################################################################################
# outputFolder=${outputDir}/control_elastic
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Elastic_1d.dat"
# path_to_input_file_3d="input_file/input_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

# ###################################################################################################################################################
# # BRAIN VISCOUS
# ###################################################################################################################################################
# outputFolder=${outputDir}/brain_viscous
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Visco_1d.dat"
# path_to_input_file_3d="input_file/input_brain_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

# ###################################################################################################################################################
# # BRAIN K VISCOUS
# ###################################################################################################################################################
# outputFolder=${outputDir}/brain_k_viscous
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Visco_1d.dat"
# path_to_input_file_3d="input_file/input_brain_k_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

###################################################################################################################################################
# BRAIN ELASTIC
###################################################################################################################################################
outputFolder=${outputDir}/brain_elastic
#remove outpur dir if it exists already
if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# create output directory
mkdir ${outputFolder}
mkdir ${outputFolder}/3D
mkdir ${outputFolder}/1D
# create the input files as copy of the originals directly in the output folder
path_to_input_file_1d="input_file/input_Elastic_1d.dat"
path_to_input_file_3d="input_file/input_brain_3d.prm"

# modify the output directory and dt in the new input files
sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
#execute command
mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
process_id=$!
wait $process_id
echo "completed " $outputFolder

# ###################################################################################################################################################
# # BRAIN K ELASTIC
# ###################################################################################################################################################
# outputFolder=${outputDir}/brain_k_elastic
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Elastic_1d.dat"
# path_to_input_file_3d="input_file/input_brain_k_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

# ###################################################################################################################################################
# # LIVER VISCOUS
# ###################################################################################################################################################
# outputFolder=${outputDir}/liver_viscous
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Visco_1d.dat"
# path_to_input_file_3d="input_file/input_liver_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

# ###################################################################################################################################################
# # LIVER K VISCOUS
# ###################################################################################################################################################
# outputFolder=${outputDir}/liver_k_viscous
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Visco_1d.dat"
# path_to_input_file_3d="input_file/input_liver_k_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

# ###################################################################################################################################################
# # LIVER ELASTIC
# ###################################################################################################################################################
# outputFolder=${outputDir}/liver_elastic
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# # create the input files as copy of the originals directly in the output folder
# path_to_input_file_1d="input_file/input_Elastic_1d.dat"
# path_to_input_file_3d="input_file/input_liver_3d.prm"

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
# process_id=$!
# wait $process_id
# echo "completed " $outputFolder

###################################################################################################################################################
# LIVER K ELASTIC
###################################################################################################################################################
outputFolder=${outputDir}/liver_k_elastic
#remove outpur dir if it exists already
if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# create output directory
mkdir ${outputFolder}
mkdir ${outputFolder}/3D
# create the input files as copy of the originals directly in the output folder
path_to_input_file_1d="input_file/input_Elastic_1d.dat"
path_to_input_file_3d="input_file/input_liver_k_3d.prm"

# modify the output directory and dt in the new input files
sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
#execute command
mpirun -np 2 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 9 >& ${outputFolder}/log
process_id=$!
wait $process_id
echo "completed " $outputFolder

