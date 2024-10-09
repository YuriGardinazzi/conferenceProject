import argparse
import os
from lm_eval.models.huggingface import HFLM
from transformers import LlamaForCausalLM,AutoTokenizer
import post_process
import lm_eval
from post_process import POSTPROCESS
import json
import numpy as np
import torch

def check_dir(path, model_name):

    folder = '.'+os.sep+path + os.sep + model_name

    if os.path.isdir(folder):
        print("Exists")
    else:
        print("Folder created")
        os.makedirs(folder,exist_ok=True)

    return

def p_column_knn(betti,ph_sim_betti):
    n_layers = len(betti)
    assert n_layers == int(ph_sim_betti.shape[0])
    tmp = np.zeros((n_layers,n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            tmp[i][j] = ph_sim_betti[i,j]/betti[j]
    
    res = np.mean(tmp,axis=1)
    return res


def layers_to_cut_given_r(r,p=0.9,layer_norm_added=True):
    '''
    r = result of the function p_column_knn
    p = value of r above which we cut layers
    '''
    res = []
    maximum = max(r)
    for i in range(len(r)-1*layer_norm_added):
        if r[i] >= p*maximum:
            res.append(i)
    return res

def full_benchmark(output_path,model_path,model_name,tasks):

    access_token =  "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"
    model = HFLM(pretrained=model_path,\
                 tokenizer=model_path,\
                 parallelize=True,\
                 device_map_option="auto", token=access_token)
    
    results = lm_eval.simple_evaluate(model=model,\
                     tasks= tasks,\
                    device="auto",\
                    batch_size="auto",\
                    num_fewshot=5)
    
    out_file = output_path + os.sep + model_name +"_"+tasks[0] + "full_bmk.json"
    with open(out_file,'w',encoding='utf-8') as f:
            json.dump(results['results'],f,ensure_ascii=False,indent=4)

    return



def load_and_cut(layers,model):

    new_layers = torch.nn.ModuleList()

    remove_layers = layers
    count = 0
    #len_layers = len(model._model.model.layers)
    len_layers = len(model._model.model.layers)
            # 32 is the number of layers of llama2
    for i in range(0, len_layers):
        if i not in remove_layers:
            #layer = model._model.model.layers[i]
            layer = model._model.model.layers[i]
            print("DIR:",dir(layer))
            layer.layer_idx = count
            layer.self_attn.layer_idx=count
            #layer.self_attn.layer_idx = count
            new_layers.append(layer)
            count += 1
    model._model.model.layers = new_layers
    changed_num_hidden_layers = len_layers - len(remove_layers)
    model.config.num_hidden_layers = changed_num_hidden_layers


    return model  

def benchmark_with_cuts(output_path,model_path,tasks,betti_path,zigzag_layers,model_name,begin_block=0,end_block=0):

    pds = post_process.read_pd_from_csv(betti_path, max_dim = 5)
    pp = POSTPROCESS(pds = pds, num_layers = zigzag_layers, start_ind = 1, zigzag = True, debug = False)

    #find betti number and phsim

    betties = pp.find_betti_layers()    
    b1 = betties[1]
    psim_mats = pp.find_ph_sim()
    psim_1 = psim_mats[1]


    for p in [0.8,0.9]:
            
        r = p_column_knn(b1,psim_1)
        list_layers = layers_to_cut_given_r(r,p=0.9)
        
        cut_path = output_path+os.sep+'cuts_'+model_name+str(p)+'.json'
        with open(cut_path,"w") as f:
            json.dump(list_layers,f,indent=4)

        print(f"BMK with {p} as threshold")

        r = p_column_knn(b1,psim_1)
        list_layers = layers_to_cut_given_r(r,p=p)
        print("LAYERS BEING CUT:\n",list_layers)
        print(f"{begin_block} - {end_block}")
        access_token = "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"
        my_model = HFLM(pretrained=model_path,\
                        tokenizer=model_path,\
                        parallelize=True,\
                        device_map_option="auto",token=access_token)
        if (begin_block > 0) and (end_block > 0):
            list_layers = [i for i in range(int(begin_block),int(end_block)+1)]  
            print("INPUT LAYERES TO CUT:\n",list_layers)
        else:
            print("STANDARD CUT")
        print("ACTUAL CUT: ",list_layers)
        res_model = load_and_cut(layers = list_layers,model=my_model)
        results = lm_eval.simple_evaluate(model=res_model,\
                                          tasks= tasks,\
                                          device="auto",\
                                          batch_size="auto",\
                                          num_fewshot=5)

        bmk_path = output_path + os.sep + model_name + '_threshold_'+str(p)+tasks[0]+'.json'
        if (begin_block > 0) and (end_block >  0):
            bmk_path = output_path + os.sep + model_name+'_'+tasks[0] +'_'+ str(begin_block) + '_' + str(end_block)+'.json'
        with open(bmk_path,'w',encoding='utf-8') as f:
            json.dump(results['results'],f,ensure_ascii=False,indent=4)
        del res_model
        if (begin_block > 0) and (end_block > 0):
            break #no need to cycle

    return
if __name__ == "__main__":
    print("BENCHMARK - v.0.1")
    #print("The following benchmark")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model to load')
    parser.add_argument('--output_path', help='path of the model to load')
    parser.add_argument('--model_name',help="name of the model run")
    parser.add_argument('--zigzag_layers', help='number of layers with intersection layers',type=int)
    parser.add_argument('--betti_path',help='folder of the output file of zigzag')
    parser.add_argument('--tasks',help="benchmarks to do")
    parser.add_argument('--full', help='Boolean value to run the full benchmarks without cuts')
    parser.add_argument('--begin',help="Beginning of the block of layers to cut",default=None,type=int)
    parser.add_argument('--end', help="End of the block of layers to cut",default=None,type=int)
    args = parser.parse_args()

    assert args.model_path != None and args.output_path != None and args.zigzag_layers != None and args.model_name != None and args.betti_path != None and args.tasks != None

    print("running benchmark with the follow parameters:\n",args)

    check_dir(path=args.output_path,\
              model_name=args.model_name)
    final_tasks = [args.tasks]    
    final_out_path = '.'+os.sep+args.output_path + os.sep + args.model_name

    print(f"args: {args.full} - {type(args.full)}")
    if args.full=='T':
        print(f"RUNNING FULL BENCHMARK {args.full}-{type(args.full)}")
        full_benchmark(model_path = args.model_path,\
                        model_name = args.model_name,\
                        output_path=final_out_path,\
                        tasks=final_tasks)
        exit()
    else:
        print("RUNNING CUTS")
        begin_block = args.begin
        end_block = args.end
        print(f"Start {begin_block} End {end_block}")
        benchmark_with_cuts(output_path=final_out_path,\
                        model_path=args.model_path,\
                        tasks=final_tasks,\
                        betti_path=args.betti_path,\
                        zigzag_layers=args.zigzag_layers,\
                        model_name=args.model_name,\
                        begin_block=int(begin_block),\
                        end_block=int(end_block))

