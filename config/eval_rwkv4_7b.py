batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'RWKV/rwkv-4-14b-pile'
dtype = 'float32' # v100 doesn't support bf16 and fp16 would overflow