#!/bin/bash
function makehostfile() {
    perl -e '@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
             print map { "$_ slots=$ENV{SLURM_GPUS_PER_NODE}\n" } @nodes;'
}
makehostfile > myhostfile
