IMG=dealii/dealii:master-focal
docker run  --user $(id -u):$(id -g) \
    --rm -t \
    -v `pwd`:/builds/app $IMG /bin/sh -c "cd /builds/app; $@"
