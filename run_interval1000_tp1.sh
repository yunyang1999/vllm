#!/usr/bin/env bash
# The same cadence test on SGLang's own shape: TP1 / DP16 across two nodes.
#
# This is the configuration where L1 does *not* flatten the load at
# step_interval=60 -- baseline imbalance 1.481 -- so if re-planning less often
# leaves LPLB more to recover anywhere, it should show here most clearly.
set -uo pipefail
H=/root/yuny/l3lab/vllm_probe
export BACKEND=allgather_reducescatter PROMPTS=512 WARMUP=2 ROUNDS=4
export TP=1 DP_PER_NODE=8 OUTDIR=interval1000_tp1 STEP_INTERVAL=1000
mkdir -p $H/results/interval1000_tp1

for DS in gsm8k gpqa; do
    for RED in 16 32; do
        bash $H/run_2node.sh eplb      $RED "$DS" "${DS}_red${RED}_baseline"
        bash $H/run_2node.sh eplb_lplb $RED "$DS" "${DS}_red${RED}_lplb"
    done
done
echo "INTERVAL1000_TP1_DONE" | tee -a $H/results/interval1000_tp1/progress.log
