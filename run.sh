sudo chmod 777 /dev/ttyUSB0

uv run scripts/compute_norm_stats.py --config-name pi0_rokae_low_mem_finetune

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_rokae_low_mem_finetune --exp-name=new_pi0_rokae_low_mem_finetune --overwrite

CUDA_VISIBLE_DEVICES=0 \
uv run scripts/serve_policy.py policy:checkpoint \
--policy.config=pi0_rokae_low_mem_finetune \
--policy.dir=checkpoints/pi0_rokae_low_mem_finetune/new_pi0_rokae_low_mem_finetune/5999

uv run scripts/compute_norm_stats.py --config-name pi05_rokae_low_mem_finetune

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_rokae_low_mem_finetune --exp-name=new_pi05_rokae_low_mem_finetune --overwrite

CUDA_VISIBLE_DEVICES=0 \
uv run scripts/serve_policy.py policy:checkpoint \
--policy.config=pi05_rokae_low_mem_finetune \
--policy.dir=checkpoints/pi05_rokae_low_mem_finetune/new_pi05_rokae_low_mem_finetune/5999
