from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset,concatenate_datasets
from datetime import datetime
import sys
import os
import numpy as np
from intrinsic_dimension.extract_activations import extract_activations
from utils.dataloader_utils import get_dataloader
from utils.dataset_utils import get_text_dataset
from utils.helpers import get_target_layers_pythia
from intrinsic_dimension.compute_distances import get_embdims



if __name__ == "__main__":
    print("****GENERATING REPS FOR PYTHIA****")
    OUTPUT_DIR = "./test2_sst"




    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-6.9b-deduped",
        revision="step143000",
        cache_dir="./pythia-6.9b-deduped/step143000",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-6.9b-deduped",
        revision="step143000",
        cache_dir="./pythia-6.9b-deduped/step143000",
    )

    raw_dataset_tmp = load_dataset("sst","default")
    
    #raw_dataset_tmp = load_dataset("NeelNanda/pile-10k",trust_remote_code=True)
    raw_dataset = concatenate_datasets([raw_dataset_tmp['train'],\
                                        raw_dataset_tmp['validation'],\
                                        raw_dataset_tmp['test']]) 
    
    
    dataset = get_text_dataset(text_field='sentence', tokenizer=tokenizer, raw_dataset=raw_dataset)
    
    nsamples = dataset.num_rows
    dataloader = get_dataloader(dataset=dataset,pad_token_id=2,batch_size=1,max_seq_len=512)
    
    target_layers = get_target_layers_pythia(model,32,option="norm2")
    target_layers = [x for x in target_layers.values()]

    print(f"TARGET_LAYERS: {target_layers}")
    embdim, dtypes = get_embdims(model,dataloader,target_layers)
    
    EA = extract_activations(model,model_name='llama2-70b',dataloader=dataloader,target_layers=target_layers, embdim=embdim,nsamples=nsamples,dtypes=dtypes,use_last_token=True)
    EA.extract(dataloader)

    if not os.path.isdir(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR) 

    for name in target_layers:
        np.save(f"{OUTPUT_DIR}/rep-{name}.npy",EA.hidden_states[name])
