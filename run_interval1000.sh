#!/usr/bin/env bash
# Does LPLB's trend across redundancy flip when the placement is re-planned at
# SGLang's cadence instead of mine?
#
# Every number in this project so far used step_interval=60. That was my choice
# early on, to make rearrangements frequent enough to observe, and it then
# stayed. It is 50x more often than vLLM's own default of 3000 and 17x more
# often than SGLang's 1000 -- and re-planning that often is what keeps L1
# tracking the live load, which is why our baseline sits at 1.03-1.16 and LPLB
# finds nothing left to recover.
#
# So this is not a tuning sweep. It tests whether "our L1 already did the work"
# is a property of the algorithms or an artefact of how often I ran it.
set -uo pipefail
H=/host/l3lab/vllm_probe
OUT=$H/results/interval1000
mkdir -p "$OUT"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export MATHDX_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/mathdx
export MLB_MOE_BACKEND=deep_gemm
export PYTHONPATH=/host/moe_load_balancer/src
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

cell() {
    local name=$1 mode=$2 ds=$3 red=$4
    [ -f "$OUT/${name}.json" ] && { echo "== skip $name"; return 0; }
    echo "=== STARTING $name $(date -u +%H:%M:%S) ===" | tee -a "$OUT/progress.log"
    timeout 5400 python3 $H/bench_cell_dp.py \
        --dataset "$H/datasets/${ds}_2000x1024.json" --tp 2 --dp 4 \
        --prompts 512 --warmup 2 --rounds 4 --max-num-batched-tokens 4096 \
        --redundant "$red" --gpu-util 0.92 --step-interval 1000 \
        --all2all-backend allgather_reducescatter --eager --mode "$mode" \
        --out "$OUT/${name}.json" > "$OUT/${name}.log" 2>&1
    local rc=$?
    python3 $H/merge_balance.py "$OUT/${name}.log" "$OUT/${name}.json" 2>/dev/null
    echo "=== ${name} EXIT=$rc $(date -u +%H:%M:%S) ===" | tee -a "$OUT/progress.log"
}

for DS in gsm8k gpqa; do
    for RED in 16 32; do
        cell "${DS}_red${RED}_baseline" eplb      "$DS" $RED
        cell "${DS}_red${RED}_lplb"     eplb_lplb "$DS" $RED
    done
done
echo "INTERVAL1000_DONE" | tee -a "$OUT/progress.log"
