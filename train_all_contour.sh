#!/bin/bash -x

# This script reproduces the experiments for Figure 8 and Table 1 in the paper.

set -e

trap exit SIGINT

ulimit -v 16000000000

common="jbsub -mem 8g -cores 1+1 -queue x86_6h -proj $(date +%Y%m%d%H%M)"

export PYTHONUNBUFFERED=1

# A=1-4, U=1-20, P=1-20
parallel $common ./strips.py learn_plot puzzle FirstOrderAE mnist 3 3 {} 20000 ::: $(seq 1 20) ::: $(seq 1 4) ::: $(seq 1 20)

# A=9 (see all args), U=1, P=1-400
parallel $common ./strips.py learn_plot puzzle FirstOrderAE mnist 3 3 {} 20000 ::: 1 ::: 9 ::: $(seq 1 400)

common="jbsub -mem 32g -cores 1 -queue x86_1h -proj $(date +%Y%m%d%H%M)"

# parallel "$common 'lisp/domain-fol.bin {} 9 samples/puzzle_mnist_3_3_{1}_{2}_{3}_20000_p4/all_actions_as_predicates.csv > samples/puzzle_mnist_3_3_{1}_{2}_{3}_20000_p4/domain.pddl'" \
#          ::: $(seq 1 20) ::: $(seq 1 4) ::: $(seq 1 20)
