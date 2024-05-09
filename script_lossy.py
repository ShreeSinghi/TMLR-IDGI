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
    n_images = sum(len(x) for x in class_file_dict)

    if opt.method == "standard":
        integrator = IntegratedGradients(model, preprocess, load)
    if opt.method == "blurig":
        integrator = BlurIG(model, preprocess, load)
    if opt.method == "gig":
        integrator = GIG(model, preprocess, load)
    
    compute_at = [opt.n_steps]

    loss_ig = 0
    loss_idgi = 0

    for i in tqdm(range(opt.resume_epoch, 1000, CLASS_CLUSTER)):
        print("Processing classes", i, "to", i+CLASS_CLUSTER)
        image_paths = []
        indexs = []
        for class_idx in range(i, i+CLASS_CLUSTER):
            image_paths.extend([os.path.join(opt.dataroot, "val", "lossy", file ) for file in class_file_dict[class_idx]])
            indexs += [class_idx] * len(class_file_dict[class_idx])
        
        salienciency_ig, salienciency_idgi = integrator.saliency(image_paths, indexs, opt.n_steps, compute_at, batchSize)
        pointer = 0
        for class_idx in range(i, i+CLASS_CLUSTER):
            class_size = len(class_file_dict[class_idx])

            for j, n_step in enumerate(compute_at):
                ig_clean   = np.load(os.path.join(opt.dataroot, "saliencies", opt.model,   f"{opt.method}_ig", f"class_{class_idx:03d}_steps_{n_step}.npy"))
                idgi_clean = np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_idgi", f"class_{class_idx:03d}_steps_{n_step}.npy"))
                ig_clean /= ig_clean.sum((1,2), keepdims=True)
                idgi_clean /= idgi_clean.sum((1,2), keepdims=True)
                
                ig_lossy   = salienciency_ig[j][pointer:pointer+class_size]
                idgi_lossy = salienciency_idgi[j][pointer:pointer+class_size]
                ig_lossy /= ig_lossy.sum((1,2), keepdims=True)
                idgi_lossy /= idgi_lossy.sum((1,2), keepdims=True)

                loss_ig += np.square(ig_clean - ig_lossy).sum()
                loss_idgi += np.square(idgi_clean - idgi_lossy).sum()
                print(loss_ig /(n_images*np.prod(ig_clean.shape[1:])), loss_idgi /(n_images*np.prod(ig_clean.shape[1:])))

            pointer += class_size
        del salienciency_ig, salienciency_idgi
        gc.collect()

    print("MSE IG:  ", loss_ig  /(n_images*np.prod(ig_clean.shape[1:])))
    print("MSE IDGI:", loss_idgi/(n_images*np.prod(idgi_clean.shape[1:])))