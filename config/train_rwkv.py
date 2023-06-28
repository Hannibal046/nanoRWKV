# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'nanoRWKV'
wandb_run_name='RWKV-130M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# rwkv specific parameters
dtype = 'float16' # v100 doesn't support bf16
model_type = 'rwkv'
# beta1 = 0.9
# beta2 = 0.99
# learning_rate = 8e-4
# min_lr = 1e-5
# warmup_iters = 0

weight_decay = 1e-1
use_customized_cuda_kernel = True

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

