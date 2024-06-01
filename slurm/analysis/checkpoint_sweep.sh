#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=0-02:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=checkpoint_sweep


# u0gf0sgx
# run_id="ol-state-dr-low-1/u0gf0sgx"
# randomness="low"

# ykosiypt, qzt2lh9x, frmymr5x, xaqzl3mx, 8k6oiit5, otft0k6k, 9zjnzg4r, l8avaysq
# run_id="ol-state-dr-med-1/l8avaysq"
# randomness="med"

# c24b6odm, zf8p0san, 93sr48mc, rwnni8cs
# run_id="ol-state-dr-high-1/rwnni8cs"
# randomness="high"


# 9s7hrl4i, nv48q2hd, ub3omf25, 99ao7drw
run_id="rt-state-dr-low-1/99ao7drw"
randomness="low"

# one_leg, round_table
task="round_table"

# 750, 1000
timesteps=1000

root_dir=outputs
folder=$(dirname "$root_dir/$run_id")

if [ ! -d "$folder" ]; then
    echo "Folder '$folder' does not exist. Creating it..."
    mkdir -p "$folder"
fi

csv_file="$root_dir/${run_id}_results.csv"
echo "wt_type,success_rate" > "$csv_file"

for ((i=99; i<=4999; i+=100)); do
    wt_type="_$i.pt"
    
    output=$(python -m src.eval.evaluate_model --run-id "$run_id" --n-envs 128 --n-rollouts 128 \
        -f "$task" --if-exists append --max-rollout-steps $timesteps --controller diffik --use-new-env \
        --action-type pos --observation-space state --randomness $randomness --wt-type "$wt_type")
    
    success_rate=$(echo "$output" | grep -oP "Success rate: \K[\d.]+")
    success_count=$(echo "$output" | grep -oP "Success rate: [\d.]+% \(\K\d+")
    rollout_count=$(echo "$output" | grep -oP "Success rate: [\d.]+% \(\d+/\K\d+")
    
    echo "$i,$success_rate" >> "$csv_file"
    echo "wt_type: $wt_type, Success rate: $success_rate ($success_count/$rollout_count)"
done