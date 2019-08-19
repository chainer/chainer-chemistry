#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate py27
exec "$@"
