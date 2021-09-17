Bare template for deal.II Application
=====================================

[![Build Status](https://travis-ci.org/luca-heltai/bare-dealii-app.svg)](https://travis-ci.org/luca-heltai/bare-dealii-app)

[![Build Status](https://gitlab.com/luca-heltai/bare-dealii-app/badges/master/pipeline.svg)](https://gitlab.com/luca-heltai/bare-dealii-app/)


A bare deal.II application, with directory structure, a testsuite, and unittest
block based on google tests.

This repository can be used to bootstrap your own deal.II
application. The structure of the directory is the following:

	./source
	./include
	./tests
	./gtests
	./doc

The directories contain a minimal working application (identical to step-6, 
where implementations and declarations have been separated) to solve the
Poisson problem on a square, a test directory that uses deal.II style testing, 
a test directory that uses google tests, and a doc directory, that contains
a `Doxyfile` to use with `doxygen`.

The `CMakeLists.txt` will generate both some executables and two libraries
containing all cc files **except** `source/main.cc`, one for Debug mode and
one for Release mode. This library is linked to the running tests, so that you 
can make tests on your application just as you would do with the deal.II 
library.

Modify the TARGET variable in the CMakeLists.txt to your application
name. Two libraries named ./tests/lib${TARGET}.so and ./tests/lib${TARGET}.g.so
will be generated together with one executable per dimension, per build type,
i.e., a total of six executables, and two libraries.

After you have compiled your application, you can run 

	make test

or
	
	ctest 

to start the testsuite.

Take a look at
https://www.dealii.org/developer/developers/testsuite.html for more
information on how to create tests and add categories of tests, and a look at
https://github.com/google/googletest/blob/master/googletest/docs/primer.md
for a quick setup of unit tests with google test.

Both `.travis.yml` and `.gitlab-ci.yml` files are provided that 
build the application and run the tests in the tests directory using
ctest, in continuous integration, by running under docker with the 
image provided on dockerhub.com: `dealii/dealii:master-focal`.

Moreover, three github actions are provided to check indentation, build
the documentation, and test the library from within github actions.

The documentation is built and deployed at each merge to master. You can 
find the latest documentation here:
https://luca-heltai.github.io/bare-dealii-app/

Licence
=======

See the file ./LICENSE for details
