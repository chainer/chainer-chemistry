#!/bin/bash
set -eux

BASE=6.0.0

service docker stop
mount -t tmpfs -o size=100% tmpfs /var/lib/docker
service docker start
gcloud auth configure-docker

if [ ${CHAINERX} -gt 0 ]; then
    if [ ${GPU} -gt 0 ]; then
        case ${CHAINER} in
            stable)
                DOCKER_IMAGE=asia.gcr.io/pfn-private-ci/chainer-chem-py$PYTHON-chx-gpu-stable:latest
                ;;
            latest)
                DOCKER_IMAGE=asia.gcr.io/pfn-private-ci/chainer-chem-py$PYTHON-chx-gpu-latest:latest
                ;;
            base)
                DOCKER_IMAGE=asia.gcr.io/pfn-private-ci/chainer-chem-py$PYTHON-chx-gpu-base:latest
                ;;
        esac
    else
        case ${CHAINER} in
            stable)
                DOCKER_IMAGE=asia.gcr.io/pfn-private-ci/chainer-chem-py$PYTHON-chx-cpu-stable:latest
                ;;
            latest)
                DOCKER_IMAGE=asia.gcr.io/pfn-private-ci/chainer-chem-py$PYTHON-chx-cpu-latest:latest
                ;;
            base)
                DOCKER_IMAGE=asia.gcr.io/pfn-private-ci/chainer-chem-py$PYTHON-chx-cpu-base:latest
                ;;
        esac
    fi
    echo "Use installed chainer in Docker image"
else
    DOCKER_IMAGE=asia.gcr.io/pfn-private-ci/chainer-chem-py$PYTHON:latest
    case ${CHAINER} in
        stable)
            echo pip install chainer >> install.sh
            ;;
        latest)
            echo pip install --pre chainer >> install.sh
            ;;
        base)
            echo pip install chainer==${BASE} >> install.sh
            ;;
    esac
    if [ ${GPU} -gt 0 ]; then
        case ${CHAINER} in
            stable)
                echo pip install cupy-cuda101 >> install.sh
                ;;
            latest)
                echo pip install --pre cupy-cuda101 >> install.sh
                ;;
            base)
                echo pip install cupy-cuda101==${BASE} >> install.sh
                ;;
        esac
    fi
fi

echo pip install pytest-cov pytest-xdist mock >> install.sh
echo pip install -e . >> install.sh

echo $DOCKER_IMAGE
cat install.sh

if [ ${GPU} -gt 0 ]; then
    PYTEST_OPTION="not slow"
    RUNTIME="--runtime=nvidia"
else
    PYTEST_OPTION="not slow and not gpu"
    RUNTIME=""
fi

docker run $RUNTIME --interactive --rm \
    --volume $(pwd):/repo/ --workdir /repo/\
    $DOCKER_IMAGE sh -ex << EOD
. ./install.sh
pytest -n 4 --cov=chainer_chemistry -m '${PYTEST_OPTION}' tests/
EOD
