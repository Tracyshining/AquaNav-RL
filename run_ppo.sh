#!/usr/bin/env bash
# Train independent PPO policies for no current and uniform current, then plot and animate results.
set -euo pipefail
STEPS="${1:-1000000}"
N_ENVS="${2:-4}"
SEED="${3:-7}"
RESULT_DIR="result"
RUN_DIR="${RESULT_DIR}/seed_${SEED}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

python -m aquanav.train_ppo --current none --steps "$STEPS" --n-envs "$N_ENVS" --seed "$SEED" --output-dir "$RESULT_DIR" > "${LOG_DIR}/ppo_none.log" 2>&1 &
PID_NONE=$!
python -m aquanav.train_ppo --current uniform --steps "$STEPS" --n-envs "$N_ENVS" --seed "$SEED" --output-dir "$RESULT_DIR" > "${LOG_DIR}/ppo_uniform.log" 2>&1 &
PID_UNIFORM=$!
echo "PPO jobs: no-current PID=${PID_NONE}, uniform-current PID=${PID_UNIFORM}"

FAILED=0
wait "$PID_NONE" || FAILED=1
wait "$PID_UNIFORM" || FAILED=1
if [ "$FAILED" -ne 0 ]; then
    echo "A PPO training job failed; check ${LOG_DIR}/ppo_{none,uniform}.log" >&2
    exit 1
fi

python -m aquanav.evaluate_ppo --seed "$SEED" --model-dir "$RESULT_DIR" --output "$RUN_DIR" --episodes 100
echo "Done. Models, metrics, PNG plots and both GIF animations are in ${RUN_DIR}/"
