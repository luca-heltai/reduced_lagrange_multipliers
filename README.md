Reduced Lagrange Multipliers Method
===================================

![GitHub CI](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/doxygen.yml/badge.svg)
![Indent](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/indentation.yml/badge.svg)


The documentation is built and deployed at each merge to master. You can 
find the latest documentation here:

https://luca-heltai.github.io/reduced_lagrange_multipliers/

Licence
=======

See the file ./LICENSE for details

USE
=======



run in parallel as

export OMP_NUM_THREADS=1 

mpirun -np n ./build/coupled_elasticity_debug <path_to_input_3d> <path_to_input_1d> <couplingSampling> <couplingStart> 0
if we only want the the 1D simulation then set coupling Start to 100
if we only want the 3D Simulation then only give <path_to_input_3d>

random error "invalid template argument" solved by changing the order od #include in app_*

per sensitivity and in general for bifurcation select

      set Max steps     = 10000
      set Reduction     = 1.e-8
      set Tolerance     = 1.e-14
    end
    subsection Outer control
      set Log frequency = 1
      set Log history   = false
      set Log result    = true
      set Max steps     = 10000
      set Reduction     = 1.e-8
      set Tolerance     = 1.e-14

