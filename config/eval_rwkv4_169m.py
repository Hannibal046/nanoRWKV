# evaluate the RWKV-4-169M
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
dtype = 'float16' # v100 doesn't support bf16
init_from = 'RWKV/rwkv-4-169m-pile'
