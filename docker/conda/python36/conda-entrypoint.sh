#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate py36
exec "$@"
