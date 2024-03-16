DATASET=../train/dataset

cd ./src

python3 main.py \
 --exp_id train \
 --Problem_AIG_Dir ${DATASET} \
 --RL_mode train \
 --customized_mapper ./mockturtle/build/examples/my_mapper \
 --baseline_mapper ./mockturtle/build/examples/my_baseline \
 --min_solve_time 0 --max_solve_time 100 