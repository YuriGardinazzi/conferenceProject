# %%
import torch
import numpy as np
import os
import argparse
import random
import esm

torch.cuda.empty_cache()
torch.set_grad_enabled(False)

def read(filename):
    file = open(filename,'r') 
    records = [line.strip() for line in file]
    records = records

    n_seq = len(records)
    max_len = max([len(records[i]) for i in range(n_seq)])
    if max_len > 1022:
        max_len = 1022
    
    # empirical value for memory handle il GPU 
    # with this value you are able to store all the reps
    bdim = 10000//max_len
    batch_dim = min(bdim, n_seq)

    all_batch = []
    all_len = []

    b = 0
    # iterate over batches
    while (b < n_seq-1):
        seq_list = []
        seq_len= []

        batch_dim = min(n_seq - b, batch_dim)

        # iterate over sequences
        for i in range(b, b+batch_dim):
            seq = records[i]
            
            if len(seq) > 1022:
               seq = seq[:1022]
               
            # the batch_converter needs for each sequence a tuple with an ID (starting with >)
            # and the sequence itself
            seq_list.append(('>', seq))
            seq_len.append(len(seq))
        
        b+= batch_dim
        all_batch.append(seq_list)
        all_len.append(seq_len)
        
    return all_batch, all_len
    

def inference(esm_bc, esm_t, batch, rep):
    _, _, esm_bt = esm_bc(batch)
    esm_bt = esm_bt.cuda()
    inference = esm_t(esm_bt, repr_layers = rep)
    return inference['representations']


def seq_mean(res, seq_len):
    mean_res = res[1 : seq_len + 1].mean(0)
    return mean_res


if __name__ == "__main__":

    print('Code starts: inputs')
    parser = argparse.ArgumentParser()
    
    # in this case the input is a file with one sequence per line
    # each sequence is already without gap (-) and all in capital letters
    parser.add_argument("-input", "--inputdir", default = 'proteinnet_train', type = str, help = "inputdir")
    parser.add_argument("-output", "--outputdir", default = '', type = str, help = "outputdir")    
    # you can add other parser for example the model name, the number of reps to extract, cuda running mode

    args = parser.parse_args()
    input_file = args.inputdir
    res_dir = args.outputdir    
    #os.makedirs(res_dir, exist_ok = True)
         
    random.seed(609)
    
    print('Executing esm')
    # you can choose between different models 
    esm, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm.eval()#.cuda()
    
    print('executing batch converter')
    # the same for every model
    esm_batch_converter = esm_alphabet.get_batch_converter()
    

    n_reps = np.arange(33) #from 0 (embedding layer) to n of layers of the choosen model 

    reps = [i for i in n_reps]
    print('reps:')
    print(reps)    
    # store all the mean 
    all_mean =  {k: [] for k in reps} 
    
    # retrieve the batches and the len of seq inside each batch
    seq_list, seq_len = read(input_file)   

    
    print('now calculate inference over each batch')
    # iterate over each batch 
    for idx_b, batch in enumerate(seq_list):
         res = inference(esm_batch_converter, esm, batch, reps) 
          
         print('iterating sequences inside this batch')
         # iterates over each sequence inside the batch
         for idx_s, seq in enumerate(batch):
             
             print('iterating on every reps')
             # iterate on every reps
             for r in reps:
                 all_mean[r].append(seq_mean(res[r][idx_s].cpu().detach().numpy(), seq_len[idx_b][idx_s]))
                 
    print('saving')
    for r in all_mean.keys():
         reps_mean = np.stack(all_mean[r])
         mean_name = os.path.join(res_dir, "rep-"+str(r)+".npy")
         np.save(mean_name,  reps_mean)



