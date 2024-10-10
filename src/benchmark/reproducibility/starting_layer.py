import argparse

from datasets import load_dataset
from short_transformers import ShortTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from short_transformers.dist import bi_score, angular_distance_all_tokens
from short_transformers.utils import (
    draw_diagram,
    get_scored_blocks,
    get_best_pruning_start,
)

from pythia_version.short_transformers.short_transformer import ShortTransformer as PythiaShortTransformer
from pythia_version.short_transformers.dist import pythia_bi_score, pythia_angular_distance_all_tokens
from pythia_version.short_transformers.utils import (
    get_scored_blocks as pythia_get_scored_blocks,
    get_best_pruning_start as pythia_get_best_pruning_start,
)

def run_pythia(access_token,model_name,block_number,method):
    
    model = PythiaShortTransformer.from_pretrained(model_name, device_map="auto",token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("NeelNanda/pile-10k",split="train",trust_remote_code=True)

    if method == "biscore":
        model.set_metric(pythia_bi_score)
    else:
        model.set_metric(pythia_angular_distance_all_tokens)
    
    results = model.analyse_layers(
        dataset=dataset,
        tokenizer=tokenizer,
        use_chat_template=False,
        key="text",
        limit=100,
        max_length=1000,
    )
    start_layer = pythia_get_best_pruning_start(results, block_size=block_number)
    block_score = pythia_get_scored_blocks(results, return_md=True)
    print("results", results)
    print("block_score: ",block_score)
    print("best: ",start_layer)

    return



def run(access_token,model_name,block_number,method):
    
    model = ShortTransformer.from_pretrained(model_name, device_map="auto",token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("NeelNanda/pile-10k",split="train",trust_remote_code=True)

    if method == "biscore":
        model.set_metric(bi_score)
    else:
        model.set_metric(angular_distance_all_tokens)
    
    results = model.analyse_layers(
        dataset=dataset,
        tokenizer=tokenizer,
        use_chat_template=False,
        key="text",
        limit=100,
        max_length=1000,
    )
    start_layer = get_best_pruning_start(results, block_size=block_number)
    block_score = get_scored_blocks(results, return_md=True)
    print("results", results)
    print("block_score: ",block_score)
    print("best: ",start_layer)

    return

if __name__ == "__main__":
    model_list = ['meta-llama/Llama-2-7b-hf',
                  'meta-llama/Llama-2-13b-hf',
                  'meta-llama/Llama-2-70b-hf',
                  'meta-llama/Meta-Llama-3-8B',
                  'meta-llama/Meta-Llama-3-70B',
                  'EleutherAI/pythia-6.9b-deduped',
                  'mistralai/Mistral-7B-v0.1']
    
    parser = argparse.ArgumentParser(description='Run and reproduce benchmarks')
    
    parser.add_argument('--model', type=str, help='input model, e.g. "meta-llama/Meta-Llama-3-8B"')
    parser.add_argument('--block number',type=int,help='Number of blocks to cut')
    parser.add_argument('--token',type=str,help='User Hugging Face token')
    parser.add_argument('--method',type=str,help='personal access token for hugging face')