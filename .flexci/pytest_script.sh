#!/bin/bash
set -eu

main() {
  py_version=$1
  prepare_docker &
  wait
  docker run --runtime=nvidia --rm --volume=$(pwd):/repo \
      asia.gcr.io/pfn-public-ci/chainer-chem-"$py_version":latest \
      bash -c "cp -r /repo .; cd repo; \
      pip install pytest-cov mock; \
      pip install -e .; \
      pytest --cov=chainer_chemistry -m 'not slow' tests/"
}

# prepare_docker makes docker use tmpfs to speed up.
# CAVEAT: Do not use docker during this is running.
prepare_docker() {
  service docker stop
  mount -t tmpfs -o size=100% tmpfs /var/lib/docker
  service docker start
  gcloud auth configure-docker
}

main "$@"
