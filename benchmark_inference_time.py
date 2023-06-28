from pathlib import Path

import pandas as pd
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from torch import nn
import torch
from time import sleep
import gc

# DATA_PATH = Path(__file__).parent / '../data'
DATA_PATH = Path("") / 'data'

LOGITS_PROCESSOR = LogitsProcessorList()

def sample(outputs):
    next_token_logits = outputs.logits[:, -1, :]
    probs = nn.functional.softmax(next_token_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_tokens

devices = ['cuda'] # , 'cpu']
recompute_all_models = True
models = [
    # Bloom
    # "EleutherAI/gpt-neo-125m",
    # "facebook/opt-125m",
    # "EleutherAI/pythia-160m",
    # "facebook/opt-350m",
    # "EleutherAI/pythia-410m",
    # "bigscience/bloom-560m",

    # # OPT 
    # "facebook/opt-1.3b",
    # "bigscience/bloom-1b7",
    # "EleutherAI/gpt-neo-1.3B",
    # "EleutherAI/pythia-1.4b",

    # # GPT-NEO
    # "facebook/opt-2.7b",
    # "EleutherAI/gpt-neo-2.7B",
    # "EleutherAI/pythia-2.8b",
    # "bigscience/bloom-3b",

    # # pythia
    # "facebook/opt-6.7b",
    "EleutherAI/pythia-6.9b",
    # "facebook/opt-13b",
    # "EleutherAI/pythia-12b",

]
num_tokens = 1024
num_samples = 1
prompt = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'

data = []
if (DATA_PATH / 'inference_results_hf.csv').exists() and not recompute_all_models:
    data = pd.read_csv(DATA_PATH / 'inference_results_hf.csv').to_dict('records')

for device in devices:
    for model_name in tqdm(models):
        print(model_name)
        if any(d["model_name"] == model_name and d["strategy"] == device for d in data): 
            continue

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(device)

            model_size = sum(p.numel() for p in model.parameters())

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenized_prompt = tokenizer(prompt, return_tensors="pt")
            tokenized_prompt = tokenized_prompt['input_ids'].to(device)

            for tok_idx in range(num_tokens):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
                    with record_function("model_inference"):
                        tokens = model.forward(tokenized_prompt)

                full_profile = next(event for event in prof.key_averages() if event.key == 'model_inference')
                next_tokens = sample(tokens)
                tokenized_prompt = torch.cat([tokenized_prompt, next_tokens[:, None]], dim=-1)
                gen_text = tokenizer.decode(tokenized_prompt[0])
                data.append({
                    "model_name": model_name,
                    "model_size": model_size,
                    "token_id": tok_idx,
                    "final_text": gen_text,
                    "strategy": device,
                    "cpu_time": full_profile.cpu_time,
                    "cuda_time": full_profile.cuda_time,
                    "cpu_memory_usage": full_profile.cpu_memory_usage,
                    "cuda_memory_usage": full_profile.cuda_memory_usage,
                    "self_cpu_memory_usage": full_profile.self_cpu_memory_usage,
                    "self_cuda_memory_usage": full_profile.self_cuda_memory_usage
                })

                pd.DataFrame(data).to_csv(DATA_PATH / f'inference_results_hf.csv')

        except:
            print(f"FAILED AT LOADING {model_name}")
    else: 
        del model
        gc.collect()
        torch.cuda.empty_cache() 
        continue
