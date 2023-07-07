import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch import nn
import torch
torch.set_float32_matmul_precision('high')
import json
from argparse import ArgumentParser

def sample(outputs):
    next_token_logits = outputs.logits[:, -1, :]
    probs = nn.functional.softmax(next_token_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_tokens

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device",default='cuda')
    parser.add_argument("--model",required=True)
    parser.add_argument("--use_cache",action='store_true')
    parser.add_argument("--max_new_tokens",type=int,default=16_000)
    parser.add_argument("--output_path")
    args = parser.parse_args()

    prompt = 'hello' ## dummpy input

    config = AutoConfig.from_pretrained(args.model)
    config.max_position_embeddings = args.max_new_tokens+10
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    model = model.to(args.device)
    model = torch.compile(model)
    model_size = sum(p.numel() for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    tokenized_prompt = tokenized_prompt['input_ids'].to(args.device)            

    model_input = {
        "input_ids":tokenized_prompt,
        "use_cache":args.use_cache,
    }

    cache_name = "state" if args.model.startswith("RWKV") else "past_key_values"
    model_input[cache_name]=None

    os.makedirs(os.path.dirname(args.output_path),exist_ok=True)
    writer = open(args.output_path,'w')
    for tok_idx in range(args.max_new_tokens):
        with torch.no_grad():
            if args.use_cache and model_input[cache_name] is not None:model_input["input_ids"] = tokenized_prompt[:,-1:].to(args.device)
            else:model_input["input_ids"] = tokenized_prompt.to(args.device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
                with record_function("model_inference"):
                    output = model.forward(**model_input)

        model_input[cache_name]=getattr(output,cache_name)
        next_tokens = sample(output)
        tokenized_prompt = torch.cat([tokenized_prompt.cpu(), next_tokens[:, None].cpu()], dim=-1)
        
        full_profile = next(event for event in prof.key_averages() if event.key == 'model_inference')
        writer.write(json.dumps({
            "model_name": args.model,
            "model_size": model_size,
            "token_id": tok_idx,
            "strategy": args.device,
            "cpu_time": full_profile.cpu_time,
            "cuda_time": full_profile.cuda_time,
            "cpu_memory_usage": full_profile.cpu_memory_usage,
            "cuda_memory_usage": full_profile.cuda_memory_usage,
            "self_cpu_memory_usage": full_profile.self_cpu_memory_usage,
            "self_cuda_memory_usage": full_profile.self_cuda_memory_usage,
            "max_memory_allocated":torch.cuda.max_memory_allocated(),
        })+'\n'
        )
        torch.cuda.empty_cache()

    writer.close()

"""
python benchmark_inference_time.py --model RWKV/rwkv-4-3b-pile --use_cache --output_path data/inference_time/rwkv-3b.jsonl
python benchmark_inference_time.py --model RWKV/rwkv-4-7b-pile --use_cache --output_path data/inference_time/rwkv-7b.jsonl
python benchmark_inference_time.py --model RWKV/rwkv-4-14b-pile --use_cache --output_path data/inference_time/rwkv-14b.jsonl
python benchmark_inference_time.py --model facebook/opt-2.7b --use_cache --output_path data/inference_time/opt-2.7b.jsonl
python benchmark_inference_time.py --model facebook/opt-6.7b --use_cache --output_path data/inference_time/opt-6.7b.jsonl
python benchmark_inference_time.py --model EleutherAI/pythia-2.8b --use_cache --output_path data/inference_time/pythia-2.8b.jsonl
python benchmark_inference_time.py --model EleutherAI/pythia-6.9b --use_cache --output_path data/inference_time/pythia-6.9b.jsonl
python benchmark_inference_time.py --model EleutherAI/gpt-neo-2.7B --use_cache --output_path data/inference_time/gpt-neo-2.7B.jsonl

############# Poltting Code ##############
import numpy as np
import json
def get_jsonl(f): return [json.loads(x) for x in open(f).readlines()]
import matplotlib.pyplot as plt
fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(18, 4))

for model_name in [
    "rwkv-3b",
    # "rwkv-7b",
    # "rwkv-14b",
    "opt-2.7b",
    "gpt-neo-2.7B",
    "pythia-2.8b"
    ]:
    data = get_jsonl(f"data/inference_time/{model_name}.jsonl")
    cuda_time = [x['cuda_time'] for x in data]
    cumulative_time = np.cumsum(cuda_time)/(1000*1000)
    memory_usage = [x['max_memory_allocated']/(2**10)/(2**10)/(2**10) for x in data]
    ax1.plot([x/1000 for x in cuda_time][100:],label=model_name)
    ax2.plot(cumulative_time,label=model_name)
    ax3.plot(memory_usage,label=model_name)

ax1.set_xlabel("# Tokens")
ax1.set_ylabel("Time (ms) to generated the #-th token")
ax1.grid()
ax1.legend()
ax1.set_title("Single Token Generation Latency")

ax2.set_xlabel("# Tokens")
ax2.set_ylabel("Cumulative time (s) to generated the #-th token")
ax2.grid()
ax2.legend()
ax2.set_title("Cumulative Generation Latency")

ax3.set_xlabel("# Tokens")
ax3.set_ylabel("Memory usage (GB)")
ax3.grid()
ax3.legend()
ax3.set_title("Memory usage in Generation")
"""