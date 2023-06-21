# nanoRWKV

![nanoGPT](assets/nanorwkv.jpg)

The [nanoGPT](https://github.com/karpathy/nanoGPT)-style implementation of [RWKV Language Model](https://www.rwkv.com). It is a rewrite of [RWKV-v4neo](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4neo) and [HuggingFace Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py) that aims to create a clean code base of RWKV for head-to-head comparison with GPT-series, while keeping in line with the simplicity and practicality of nanoGPT. So it could be used to train a GPT/RWKV model in this single repository. It is still an active projects and we have managed to train a RWKV model with similar parameters to GPT-2 (124M) on OpenWebText, running on a single 8XV100 32GB node. (Your sponsorship of A100 for testing bf16 would be greatly appreciated :)

## install

```
conda create -n nanoRWKV python=3.8 && conda activate nanoRWKV
conda install cuda -c nvidia/label/cuda-11.*.0 ## for loading customized cuda kernel
pip install torch numpy transformers datasets tiktoken wandb tqdm ninja
```

## preliminary
Before kicking off this project, make sure you are familiar with:

- nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs by great [Andrej Karpathy](https://karpathy.ai). You could find the code [here](https://github.com/karpathy/nanoGPT) and teaching video [here](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- RWKV Language Model: an RNN with GPT-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). The model is created by an independent researcher [Bo Peng](https://www.zhihu.com/people/bopengbopeng). You could find the official code [here](https://github.com/BlinkDL/RWKV-LM), along with its chat version [here](https://github.com/BlinkDL/ChatRWKV). For deep understanding of this model, the [paper](https://arxiv.org/abs/2305.13048) and this [tutorial](https://johanwind.github.io/2023/03/23/rwkv_details.html) would be much helpful.

## reproducing RWKV



After all set up, let's build RWKV - first tokenize the dataset (OpenWebText):

```bash
python data/openwebtext/prepare.py
```

Then train RWKV(130M) with 8XV100 32GB on one node using PyTorch Distributed Data Parallel (DDP) :

```
torchrun --standalone --nproc_per_node=8 train.py config/train_rwkv.py
```

For comparision, we also train a GPT-2 model(124M) on the same device with:

```
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

We got the results as follows:

| model | params | train loss | val loss |
| ----- | ------ | ---------- | -------- |
| GPT-2 | 124M   |            |          |
| RWKV  | 130M   |            |          |