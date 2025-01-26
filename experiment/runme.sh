set -v
for i in 1 2 3 4 5
do
    python ../sim_logic.py experiment_$i.yml
    python ../parse.py results/experiment_$i
done