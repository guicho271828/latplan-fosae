#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

export PYTHONUNBUFFERED=1

proj=$(date +%Y%m%d%H%M)-puzzle
common="jbsub -mem 64g -cores 1+1 -queue x86_6h -proj $proj -require v100"
parallel $common ./strips.py learn_plot puzzle {} \
         ::: FirstOrderAE \
         ::: mnist \
         ::: 3 \
         ::: 3 \
         ::: None \
         ::: None \
         ::: None \
         ::: 20000


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

# parallel "$common 'lisp/domain-fol.bin {} 9 samples/blocksworld_{1}_{2}_{3}_20000_p4/all_actions_as_predicates.csv > samples/blocksworld_{1}_{2}_{3}_20000_p4/domain.pddl'" \
#          ::: $(seq 1 20) ::: $(seq 1 4) ::: $(seq 1 20)

