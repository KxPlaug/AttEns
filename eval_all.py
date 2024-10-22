
import os

attr_methods = [
    "fast_ig","deeplift","guided_ig","ig",
    "eg","sg","big",'sm','mfaba_smooth','agi','ampe','la'
]
models = ["inception_v3","resnet50","vgg16",'maxvit_t']
datasets = ["isa"]


cuda_flag = False

import multiprocess
all_commands = []
for attr_method in attr_methods:
    for model in models:
        for dataset in datasets:
            command = f"python eval.py --model {model} --attr_method {attr_method} --dataset {dataset} --single_softmax --eval_method before"
            all_commands.append(command)
            cuda_flag = not cuda_flag




            
pool = multiprocess.Pool(6)
pool.imap(os.system, all_commands)
pool.close()
pool.join()