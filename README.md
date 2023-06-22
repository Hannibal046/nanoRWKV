# nanoRWKV
> minimal implementation of RWKV language model following nanoGPT

![nanoGPT](assets/nanorwkv.jpg)

The [nanoGPT](https://github.com/karpathy/nanoGPT)-style implementation of [RWKV Language Model](https://www.rwkv.com). It is a rewrite of [RWKV-v4neo](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4neo) and [HuggingFace Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py) that aims to create a clean code base of RWKV for head-to-head comparison with GPT-series, while keeping in line with the simplicity and practicality of nanoGPT. So it could be used to train a GPT or RWKV model in this single repository. 

It is still an active project and we are training a RWKV model with similar size to GPT-2 (124M) on OpenWebText dataset on a single 8*V100 32GB node. To keep track of the **ongoing** experiments, please see this [wandb project](https://wandb.ai/hannibal046/nanoRWKV?workspace=user-hannibal046). (looks good for now)

![nanoGPT](assets/current_loss.png)

## install
```
conda create -n nanoRWKV python=3.8 
conda activate nanoRWKV
## replace * with your driver version
conda install cuda -c nvidia/label/cuda-11.*.0 
pip install torch numpy transformers datasets tiktoken wandb tqdm ninja
```
## preliminary
Before kicking off this project, make sure you are familiar with:

- **nanoGPT**: the simplest, fastest repository for training/finetuning medium-sized GPTs by great [Andrej Karpathy](https://karpathy.ai). Here you could find the [code](https://github.com/karpathy/nanoGPT) and the [teaching video](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- **RWKV Language Model**: an RNN with GPT-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). The model is created by an independent researcher [Bo Peng](https://www.zhihu.com/people/bopengbopeng). You could find the official [code](https://github.com/BlinkDL/RWKV-LM), along with its [chat version code](https://github.com/BlinkDL/ChatRWKV). For deeper understanding of this model, the [paper](https://arxiv.org/abs/2305.13048) and this [tutorial](https://johanwind.github.io/2023/03/23/rwkv_details.html) would be much helpful.

## reproducing RWKV

After all set up, let's build RWKV - first tokenize the dataset (OpenWebText):

```bash
python data/openwebtext/prepare.py
```

Then train RWKV(130M) with 8*V100 32GB on one node using PyTorch Distributed Data Parallel (DDP) :

```
torchrun --standalone --nproc_per_node=8 train.py config/train_rwkv.py
```

For comparision, we also train a GPT-2 model(124M) on the same device with:

```
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

We got the results as follows (check the progress on this [wandb project](https://wandb.ai/hannibal046/nanoRWKV?workspace=user-hannibal046)):

| model | params | train loss | val loss |
| ----- | ------ | ---------- | -------- |
| GPT-2 | 124M   |            |          |
| RWKV  | 130M   |            |          |

## to-do list
This is not a done project and there are a lot of cool stuffs to do:

- [ ] More code comment in [modeling_rwkv.py](modeling_rwkv.py).
- [ ] RNN mode for inference [[HF Implementation]](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py)
- [ ] rescale parameters for inference [[reference]](https://github.com/BlinkDL/RWKV-LM/blob/cca1b5e8e597cf40675882bb10b46287c844e35c/RWKV-v4neo/src/model_run.py#L31)
- [ ] loading RWKV checkpoint for evaluation(may not comparable to GPT-2 due to different tokenizer) 
- [ ] test bf16 training (Since V100 doesn't support bf16, your sponsorship of A100 for testing bf16 would be greatly appreciated :)
- [ ] maybe scale up a little bit with DeepSpeed? Not sure, since nanoGPT didn't do this.
- [ ] keep in line with the original implementaion of RWKV optimization. [[reference](https://github.com/BlinkDL/RWKV-LM/blob/cca1b5e8e597cf40675882bb10b46287c844e35c/RWKV-v4neo/src/model.py#L409)]
- [ ] More analysis about RWKV in [scaling_laws.ipynb](scaling_laws.ipynb), [transformer_sizeing.ipynb](transformer_sizeing.ipynb)

## what is more
If you want to know more about Large Language Models(LLM), please refer to this repository: [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM).