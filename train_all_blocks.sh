#!/bin/bash -x

# This script trains the FOSAE for blocksworld.

set -e

trap exit SIGINT

ulimit -v 16000000000

objs=${1:-5}
stacks=${2:-3}
prefix="blocks-$objs-$stacks"

export PYTHONUNBUFFERED=1

proj=$(date +%Y%m%d%H%M)-$prefix
common="jbsub -mem 64g -cores 1+1 -queue x86_6h -proj $proj -require v100"
parallel $common ./strips.py reproduce_plot blocksworld {} \
         ::: FirstOrderSAE \
         ::: $prefix \
         ::: None \
         ::: None \
         ::: None \
         ::: \
         10000 \
         ::: BCE5


# ccc/watch-proj $proj && {
#     proj=$(date +%Y%m%d%H%M)
#     common="jbsub -mem 64g -cores 1+1 -queue x86_6h -proj $proj"
#     parallel $common {} learn \
#          ::: ./state_discriminator3.py ./action_autoencoder.py \
#          ::: $prefix/*/
# }
# 
# ccc/watch-proj $proj && {
#     proj=$(date +%Y%m%d%H%M)
#     common="jbsub -mem 64g -cores 1+1 -queue x86_6h -proj $proj"
#     parallel $common {} learn \
#              ::: ./action_discriminator.py \
#              ::: $prefix/*/
# }

# common="jbsub -mem 32g -cores 1 -queue x86_1h -proj $(date +%Y%m%d%H%M)"

# parallel "$common 'lisp/domain-fol.bin {} 9 samples/blocksworld_{1}_{2}_{3}_10000_p4/all_actions_as_predicates.csv > samples/blocksworld_{1}_{2}_{3}_10000_p4/domain.pddl'" \
#          ::: $(seq 1 20) ::: $(seq 1 4) ::: $(seq 1 20)

