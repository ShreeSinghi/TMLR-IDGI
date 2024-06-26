import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np
import json
import gc
from tqdm import tqdm

from ig import IntegratedGradients
from blurig import BlurIG
from gig import GIG

from utils import load_preprocess, load_model, load_image_loader
cudnn.benchmark = True

def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--method', type=str, required=True, help='method name')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--resume_epoch', type=int, default=0, help='input batch size')
    parser.add_argument('--n_steps', type=int, default=128, help='number of steps')
    parser.add_argument('--class_cluster', type=int, default=1, help='number of classes to compute at one go')

    opt = parser.parse_args()
    batchSize = opt.batchSize
    CLASS_CLUSTER = opt.class_cluster
    print(f"Computing saliencies for {opt.model} using {opt.method} method")

    model = load_model(opt.model)
    preprocess = load_preprocess(opt.model)
    load = load_image_loader(opt.model)

    with open(os.path.join(opt.dataroot, f"{opt.model}_predictions.json"), 'r') as file:
        class_file_dict = json.load(file) # not really  a dict, its a list

    if opt.method == "standard":
        integrator = IntegratedGradients(model, preprocess, load)
        compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
    if opt.method == "blurig":
        integrator = BlurIG(model, preprocess, load)
        compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
    if opt.method == "gig":
        integrator = GIG(model, preprocess, load)
        compute_at = [opt.n_steps]

    for i in tqdm(range(opt.resume_epoch, 1000, CLASS_CLUSTER)):
        print("Processing classes", i, "to", i+CLASS_CLUSTER)
        image_paths = []
        indexs = []
        for class_idx in range(i, i+CLASS_CLUSTER):
            image_paths.extend([os.path.join(opt.dataroot, "val/images", file ) for file in class_file_dict[class_idx]])
            indexs += [class_idx] * len(class_file_dict[class_idx])
        
        salienciency_ig, salienciency_idgi = integrator.saliency(image_paths, indexs, opt.n_steps, compute_at, batchSize)
        pointer = 0
        for class_idx in range(i, i+CLASS_CLUSTER):
            class_size = len(class_file_dict[class_idx])
            os.makedirs(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_ig"), exist_ok=True)
            os.makedirs(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_idgi"), exist_ok=True)

            for j, n_step in enumerate(compute_at):
                np.save(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_ig", f"class_{class_idx:03d}_steps_{n_step}"), salienciency_ig[j][pointer:pointer+class_size].astype(np.float16))
                np.save(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_idgi", f"class_{class_idx:03d}_steps_{n_step}"), salienciency_idgi[j][pointer:pointer+class_size].astype(np.float16))
            pointer += class_size
                    
        del salienciency_ig, salienciency_idgi
        gc.collect()