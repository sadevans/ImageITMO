CODE_DIR=/netapp/a.gorokhova/projects/Attributes/orientation

GPUS=2,3
N_PROC=2
GPU=2

# for EXP in  50_dopdata_moto
for EXP in  56_augs3


do
echo orientation$EXP
export HYDRA_FULL_ERROR=1
# CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$CODE_DIR python -m torch.distributed.run --master_port 1257 --nproc_per_node $N_PROC $CODE_DIR/code/train.py --config-dir $CODE_DIR/configs --config-name orientation$EXP #> log 2> mistakes
# CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$CODE_DIR python -m torch.distributed.run --master_port 1252 --nproc_per_node $N_PROC $CODE_DIR/code/train.py --config-dir $CODE_DIR/configs --config-name orientation$EXP #> log 2> mistakes
CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$CODE_DIR python -m torch.distributed.run --master_port 1250 --nproc_per_node $N_PROC $CODE_DIR/code/train.py --config-dir $CODE_DIR/configs --config-name orientation$EXP #> log 2> mistakes
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$CODE_DIR python $CODE_DIR/code/eval.py --config-dir $CODE_DIR/configs --config-name orientation$EXP #> log 2> mistakes
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$CODE_DIR python $CODE_DIR/code/vis_mistakes_with_cm.py --cfg $CODE_DIR/configs/orientation$EXP.yaml --number 100 > log_$EXP 2> mistakes_$EXP
done