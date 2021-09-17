#!/bin/bash
IMG=`grep image .gitlab-ci.yml | awk '{print $2}'` 
echo $IMG

CMD=`grep -e '- ' .gitlab-ci.yml | sed 's/- //'`

docker run --rm -t -i -v `pwd`:/builds/app $IMG /bin/sh -c "cd /builds/app; $CMD" 
