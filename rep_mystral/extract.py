from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset,concatenate_datasets
from datetime import datetime
import sys
import os
import numpy as np
from intrinsic_dimension.extract_activations import extract_activations
from utils.dataloader_utils import get_dataloader
from utils.dataset_utils import get_text_dataset
from utils.helpers import get_target_layers_llama
from intrinsic_dimension.compute_distances import get_embdims



if __name__ == "__main__":
    print("****GENERATING REPS FOR PYTHIA****")
    OUTPUT_DIR = "./mistral_sst"

    access_token = "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"
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        cache_dir="./mistral",
        device_map="auto",
        token=access_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        cache_dir="./mistral",
        device_map="auto",
        token=access_token
    )

    raw_dataset_tmp = load_dataset("sst","default")
    #raw_dataset_tmp = load_dataset("NeelNanda/pile-10k",trust_remote_code=True)
    raw_dataset = concatenate_datasets([raw_dataset_tmp['train'],\
                                        raw_dataset_tmp['validation'],\
                                        raw_dataset_tmp['test']]) 
    
    
    dataset = get_text_dataset(text_field='text', tokenizer=tokenizer, raw_dataset=raw_dataset)
    
    nsamples = dataset.num_rows
    dataloader = get_dataloader(dataset=dataset,pad_token_id=2,batch_size=1,max_seq_len=512)
    
    #the naming conventions for the layer for mistral it's the same as llama
    target_layers = get_target_layers_llama(model,32,option="norm2")
    target_layers = [x for x in target_layers.values()]

    print(f"TARGET_LAYERS: {target_layers}")
    embdim, dtypes = get_embdims(model,dataloader,target_layers)
    
    EA = extract_activations(model,model_name='llama2-7b',dataloader=dataloader,target_layers=target_layers, embdim=embdim,nsamples=nsamples,dtypes=dtypes,use_last_token=True)
    EA.extract(dataloader)

    if not os.path.isdir(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR) 

    for name in target_layers:
        np.save(f"{OUTPUT_DIR}/rep-{name}.npy",EA.hidden_states[name])
