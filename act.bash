export WANDB_ROOT=/projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d/wandb_root
mkdir -p "$WANDB_ROOT"/{runs,cache,config}

export WANDB_DIR="$WANDB_ROOT/runs"
export WANDB_CACHE_DIR="$WANDB_ROOT/cache"
export WANDB_CONFIG_DIR="$WANDB_ROOT/config"


export XDG_CACHE_HOME="$WANDB_ROOT/cache"
export PYTHONPATH=$PWD/third_party:$PYTHONPATH
python -m lift3d.tools.act_policy \
  --config-name=train_recover \
  benchmark=act_offline \         
  agent=lift3d_act \
  task_name=peg_recover \
  dataloader.batch_size=16 dataloader.num_workers=8 dataloader.shuffle=true \
  train.num_epochs=10000 train.learning_rate=1e-5 train.kl_weight=10 \
  dataset_dir=/projects/surgical-video-digital-twin/datasets/act_peg_recover/1216/zarr \
  hydra.run.dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/3dact_lr1e-6_kl10 \
  evaluation.num_skip_epochs=15 \