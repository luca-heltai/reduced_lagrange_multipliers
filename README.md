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
mpirun -np 2 ./build/coupled_elasticity_debug <path_to_input_3d> <path_to_input_1d> <couplingSampling> <couplingStart>

20.11.23 it only works with 2 processors -> to be fixed
