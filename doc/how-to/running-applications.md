# Run an application

Applications are built from the entry points in `apps/`. Pass the parameter
file explicitly; do not rely on the current working directory.

```bash
./build/<executable> path/to/input.prm
```

Debug builds append `_debug` to the executable name. For example:

```bash
./build/elastic_static_debug build/tutorials/elastic_static/elastic_static.prm
```

The explicit elasticity example uses a manufactured wave solution and writes
the error table along with the VTU output:

```bash
./build/elasticity_explicit prms/prm_2d_euler_mms.prm
```

The [application reference](../reference/applications) is the authoritative
inventory of executable names, supported dimension combinations, optional
dependencies, and canonical examples. The [tutorials](../tutorials/index)
explain selected workflows rather than repeating the complete parameter
reference.
