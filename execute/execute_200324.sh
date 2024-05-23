#!/bin/bash

outputDir="./output/bifurcation_200521_b"

path_to_input_file_1d="input_file/input_Elastic_1d.dat"
path_to_input_file_3d="input_file/input_3d.prm"
path_to_data_file="data/inclusions_data_bif_2503_0.05.txt"
path_to_inclusions_file="data/inclusions_points_bif_2503_0.05.txt"

# ###################################################################################################################################################
# # UNCOUPLED ELASTIC
# ###################################################################################################################################################
# outputFolder=${outputDir}/uncoupled_elastic
# #remove outpur dir if it exists already
# if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# # create output directory
# mkdir ${outputFolder}
# mkdir ${outputFolder}/3D
# mkdir ${outputFolder}/1D

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# #execute command
# mpirun -np 12 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 11 >& ${outputFolder}/log
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
# mkdir ${outputFolder}/1D

# lambda=1
# mu=1

# # modify the output directory and dt in the new input files
# sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
# sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

# sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
# sed -i 's|\(set 3D 1D discretization\).*|set 3D 1D discretization                = 1|' $path_to_input_file_3d
# sed -i 's|\(set Data file\).*|set Data file                           = '${path_to_data_file}'|' $path_to_input_file_3d
# sed -i 's|\(set Inclusions file\).*|set Inclusions file                     = '${path_to_inclusions_file}'|' $path_to_input_file_3d
# sed -i 's|\(set Inclusions refinement\).*|set Inclusions refinement               = 10|' $path_to_input_file_3d
# sed -i 's|\(set Lame lambda\).*|set Lame lambda = '${lambda}'|' $path_to_input_file_3d
# sed -i 's|\(set Lame mu\).*|set Lame mu = '${mu}'|' $path_to_input_file_3d

# #execute command
# mpirun -np 12 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 3 11 >& ${outputFolder}/log
# #mpirun -np 12 ./build/elasticity_debug $path_to_input_file_3d >& ${outputFolder}/log

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

lambda=147
mu=0.67

# modify the output directory and dt in the new input files
sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
sed -i 's|\(set 3D 1D discretization\).*|set 3D 1D discretization                = 1|' $path_to_input_file_3d
sed -i 's|\(set Data file\).*|set Data file                           = '${path_to_data_file}'|' $path_to_input_file_3d
sed -i 's|\(set Inclusions file\).*|set Inclusions file                     = '${path_to_inclusions_file}'|' $path_to_input_file_3d
sed -i 's|\(set Inclusions refinement\).*|set Inclusions refinement               = 10|' $path_to_input_file_3d
sed -i 's|\(set Lame lambda\).*|set Lame lambda = '${lambda}'|' $path_to_input_file_3d
sed -i 's|\(set Lame mu\).*|set Lame mu = '${mu}'|' $path_to_input_file_3d

#execute command
mpirun -np 12 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 20 0 >& ${outputFolder}/log
process_id=$!
wait $process_id
echo "completed " $outputFolder

###################################################################################################################################################
# LIVER ELASTIC
###################################################################################################################################################
outputFolder=${outputDir}/liver_elastic
#remove outpur dir if it exists already
if [ -d "$outputFolder" ]; then rm -r $outputFolder; fi
# create output directory
mkdir ${outputFolder}
mkdir ${outputFolder}/3D
mkdir ${outputFolder}/1D

lambda=50
mu=2

# modify the output directory and dt in the new input files
sed -i 's|\(outDir\).*|outDir = '${outputFolder}'/1D/|' $path_to_input_file_1d
sed -i 's|\(outFile\).*|outFile = '${outputFolder}'/1D/|' $path_to_input_file_1d

sed -i 's|\(set Output directory\).*|set Output directory                   = '${outputFolder}'/3D/|' $path_to_input_file_3d
sed -i 's|\(set 3D 1D discretization\).*|set 3D 1D discretization                = 1|' $path_to_input_file_3d
sed -i 's|\(set Data file\).*|set Data file                           = '${path_to_data_file}'|' $path_to_input_file_3d
sed -i 's|\(set Inclusions file\).*|set Inclusions file                     = '${path_to_inclusions_file}'|' $path_to_input_file_3d
sed -i 's|\(set Inclusions refinement\).*|set Inclusions refinement               = 10|' $path_to_input_file_3d
sed -i 's|\(set Lame lambda\).*|set Lame lambda = '${lambda}'|' $path_to_input_file_3d
sed -i 's|\(set Lame mu\).*|set Lame mu = '${mu}'|' $path_to_input_file_3d

#execute command
mpirun -np 12 ./build/coupled_elasticity_debug $path_to_input_file_3d $path_to_input_file_1d 20 0 >& ${outputFolder}/log
process_id=$!
wait $process_id
echo "completed " $outputFolder

