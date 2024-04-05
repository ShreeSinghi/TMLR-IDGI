import argparse
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from PIL import Image
import gc
from torchvision import models, transforms
from tqdm import tqdm

import torch
from utils import load_preprocess, load_model, load_image_loader, compute_outputs
from metrics import insert_pixels, insertion_score, compute_pic_metric, ComputePicMetricError

cudnn.benchmark = True

import psutil


"""
Our input is
- Dataset
- Resume training point
- Correct images ki list (found from another script)
"""

def print_gpu_memory():
    print("Total GPU Memory:", torch.cuda.get_device_properties(0).total_memory/1024/1024/1024)
    print("Currently Allocated GPU Memory:", torch.cuda.memory_allocated(0)/1024/1024/1024)

def print_cpu_memory():
    cpu_memory_usage = psutil.virtual_memory()
    print("Total CPU Memory:", cpu_memory_usage.total/1024/1024/1024)
    print("Available CPU Memory:", cpu_memory_usage.available/1024/1024/1024)
    print("Used CPU Memory:", cpu_memory_usage.used/1024/1024/1024)
    print("Free CPU Memory:", cpu_memory_usage.free/1024/1024/1024)


def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

def load_images(dictionary, class_index):
    files = [os.path.join(opt.dataroot, r"val/images", file) for file in dictionary[class_index]]

    image_data = [preprocess(Image.open(path)) for path in files]
    image_data = np.array(image_data)

    return image_data

      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--method', type=str, required=True, help='method name')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--resume_epoch', type=int, default=0, help='input batch size')
    parser.add_argument('--n_steps', type=int, default=128, help='number of steps')

    opt = parser.parse_args()
    batchSize = opt.batchSize

    model = load_model(opt.model)
    preprocess = load_preprocess(opt.model)
    load = load_image_loader(opt.model)

    BLACK = -torch.Tensor((0.485, 0.456, 0.406))/torch.Tensor((0.229, 0.224, 0.225))

    with open(os.path.join(opt.dataroot, f"{opt.model}_predictions.json"), 'r') as file:
        class_file_dict = json.load(file) # not really  a dict, its a list

    if opt.method == "standard":
        compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
    if opt.method == "blurig":
        compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
    if opt.method == "gig":
        compute_at = [opt.n_steps]

    insertion_scoress_ig = []
    insertion_scoress_idgi = []

    insertion_scoress_ig_norm = []
    insertion_scoress_idgi_norm = []

    for class_idx in tqdm(range(opt.resume_epoch, 1000)):
        print("Processing class", class_idx)

        images = load([os.path.join(opt.dataroot, r"val/images", file ) for file in class_file_dict[class_idx]])
        blurs = load([os.path.join(opt.dataroot, r"val/blurred", file ) for file in class_file_dict[class_idx]])
        
        preprocessed_images = preprocess(images)
        preprocessed_blurs = preprocess(blurs)

        saliencies_ig   = torch.from_numpy(np.stack([np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_ig", f"class_{class_idx:03d}_steps_{n_step}.npy")) for n_step in compute_at]))
        saliencies_idgi = torch.from_numpy(np.stack([np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_idgi", f"class_{class_idx:03d}_steps_{n_step}.npy")) for n_step in compute_at]))
        indexs = torch.Tensor([class_idx] * (len(class_file_dict[class_idx]) * len(compute_at))).long()
        blacks = (torch.ones_like(preprocessed_images) * BLACK.view(1, 3, 1, 1))
        
        insertion_scores_ig = insertion_score(model, saliencies_ig.view(-1, *saliencies_ig.shape[2:]),
                                              preprocessed_images.repeat(len(compute_at), 1, 1, 1),
                                              blacks.repeat(len(compute_at),1,1,1),
                                              indexs)
    
        insertion_scores_idgi = insertion_score(model, saliencies_idgi.view(-1, *saliencies_idgi.shape[2:]),
                                                preprocessed_images.repeat(len(compute_at), 1, 1, 1),
                                                blacks.repeat(len(compute_at),1,1,1),
                                                indexs)

        insertion_scoress_ig_norm.append(insertion_scores_ig[0].reshape(len(compute_at), len(class_file_dict[class_idx])))
        insertion_scoress_idgi_norm.append(insertion_scores_idgi[0].reshape(len(compute_at), len(class_file_dict[class_idx])))

        insertion_scoress_ig.append(insertion_scores_ig[1].reshape(len(compute_at), len(class_file_dict[class_idx])))
        insertion_scoress_idgi.append(insertion_scores_idgi[1].reshape(len(compute_at), len(class_file_dict[class_idx])))

        gc.collect()

    insertion_scoress_ig = np.concatenate(insertion_scoress_ig, axis=1)
    insertion_scoress_idgi = np.concatenate(insertion_scoress_idgi, axis=1)

    insertion_scoress_ig_norm = np.concatenate(insertion_scoress_ig_norm, axis=1)
    insertion_scoress_idgi_norm = np.concatenate(insertion_scoress_idgi_norm, axis=1)

    print(opt.method, "IG MEAN INSERTION SCORE")
    print(insertion_scoress_ig.mean(axis=1))
    print(compute_at)

    print(opt.method, "IDGI MEAN INSERTION SCORE")
    print(insertion_scoress_idgi.mean(axis=1))
    print(compute_at)

    print(opt.method, "IG MEAN NORMALIZED INSERTION SCORE")
    print(insertion_scoress_ig_norm.mean(axis=1))
    print(compute_at)

    print(opt.method, "IDGI MEAN NORMALIZED INSERTION SCORE")
    print(insertion_scoress_idgi_norm.mean(axis=1))
    print(compute_at)
    
"""
        N_DIVISIONS = 25
        repeated_images = torch.from_numpy(images).repeat(len(compute_at))
        repeated_blurs  = torch.from_numpy(blurs).repeat(len(compute_at))
        linspace = np.linspace(0, images.shape[-1]*images.shape[-2], N_DIVISIONS).astype(int)
        
        ######### IG #########
        bokeh_images = insert_pixels(repeated_images,
                                     saliencies_ig.view(-1, *saliencies_ig.shape[2:]),
                                     repeated_blurs,
                                     linspace)
        pred_probas, pred_accs = compute_outputs(model, preprocess(bokeh_images), indexs.repeat(N_DIVISIONS))

        curve
        for i in range(len(repeated_images)):
            try:
                curve_y_compression = compute_pic_metric(repeated_images[i], pred_probas[i], bokeh_images[i], "compression")
            except ComputePicMetricError:
                curve_y_compression = compute_pic_metric(repeated_images[i], pred_probas[i], bokeh_images[i], "msssim")



        bokeh_images = insert_pixels(repeated_images,
                                     saliencies_idgi.view(-1, *saliencies_ig.shape[2:]),
                                     repeated_blurs,
                                     linspace)
        
        os.makedirs(os.path.join(opt.dataroot, "saliencies", opt.model,   f"{opt.method}_ig"), exist_ok=True)
        os.makedirs(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_idgi"), exist_ok=True)

        for j, n_step in enumerate(compute_at):
            np.save(os.path.join(opt.dataroot, "saliencies", opt.model,   f"{opt.method}_ig", f"class_{class_idx:03d}_steps_{n_step}"), salienciency_ig[j])
            np.save(os.path.join(opt.dataroot, "saliencies", opt.model,   f"{opt.method}_idgi", f"class_{class_idx:03d}_steps_{n_step}"), salienciency_idgi[j])
"""