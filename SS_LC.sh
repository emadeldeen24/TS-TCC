exp=$1
run=$2
seed=$3
dataset=$4
device=$5

python main.py --experiment_description $exp --run_description $run --seed $i --selected_dataset $dataset --device $device --training_mode "self_supervised"
python main.py --experiment_description $exp --run_description $run --seed $i --selected_dataset $dataset --device $device --training_mode "train_linear"