#!/bin/bash
source /project/rpp-tanaka-ab/hk_software/nuPRISM/sourceme.sh
module load python/2.7.14
module load scipy-stack
export PYTHONPATH=$ROOTSYS/../bindings/pyroot:$PYTHONPATH