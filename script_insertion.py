import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np
import json
from PIL import Image
import gc
from tqdm import tqdm

import torch
from utils import load_preprocess, load_model, load_image_loader, compute_outputs
from metrics import insertion_score

cudnn.benchmark = True

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
    print(f"Computing insertion score for {opt.model} using {opt.method} method")

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
        
        preprocessed_images = preprocess(images)

        saliencies_ig   = torch.from_numpy(np.stack([np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_ig", f"class_{class_idx:03d}_steps_{n_step}.npy")) for n_step in compute_at]))
        saliencies_idgi = torch.from_numpy(np.stack([np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_idgi", f"class_{class_idx:03d}_steps_{n_step}.npy")) for n_step in compute_at]))
        indexs = torch.Tensor([class_idx] * (len(class_file_dict[class_idx]) * len(compute_at))).long()
        blacks = (torch.ones_like(preprocessed_images) * BLACK.view(1, 3, 1, 1))
        
        insertion_scores_ig = insertion_score(model, saliencies_ig.view(-1, *saliencies_ig.shape[2:]),
                                              preprocessed_images.repeat(len(compute_at), 1, 1, 1),
                                              blacks.repeat(len(compute_at),1,1,1),
                                              indexs,
                                              batchSize=batchSize)
    
        insertion_scores_idgi = insertion_score(model, saliencies_idgi.view(-1, *saliencies_idgi.shape[2:]),
                                                preprocessed_images.repeat(len(compute_at), 1, 1, 1),
                                                blacks.repeat(len(compute_at),1,1,1),
                                                indexs,
                                                batchSize=batchSize)

        insertion_scoress_ig_norm.append(insertion_scores_ig[0].reshape(len(compute_at), len(class_file_dict[class_idx])))
        insertion_scoress_idgi_norm.append(insertion_scores_idgi[0].reshape(len(compute_at), len(class_file_dict[class_idx])))

        insertion_scoress_ig.append(insertion_scores_ig[1].reshape(len(compute_at), len(class_file_dict[class_idx])))
        insertion_scoress_idgi.append(insertion_scores_idgi[1].reshape(len(compute_at), len(class_file_dict[class_idx])))

        gc.collect()

    insertion_scoress_ig = np.concatenate(insertion_scoress_ig, axis=1)
    insertion_scoress_idgi = np.concatenate(insertion_scoress_idgi, axis=1)

    insertion_scoress_ig_norm = np.concatenate(insertion_scoress_ig_norm, axis=1)
    insertion_scoress_idgi_norm = np.concatenate(insertion_scoress_idgi_norm, axis=1)

    os.makedirs("results", exist_ok=True)
    with open(f"results/insertion_{opt.model}_{opt.method}_ig_{opt.n_steps}.txt", 'w') as file:
        file.write(f"IG MEAN INSERTION SCORE {insertion_scoress_ig.mean(axis=1)}\n")
        file.write(f"IG MEAN NORMALIZED INSERTION SCORE {insertion_scoress_ig_norm.mean(axis=1)}\n")
        file.write(str(compute_at))

    with open(f"results/insertion_{opt.model}_{opt.method}_idgi_{opt.n_steps}.txt", 'w') as file:
        file.write(f"IDGI MEAN INSERTION SCORE {insertion_scoress_idgi.mean(axis=1)}\n")
        file.write(f"IDGI MEAN NORMALIZED INSERTION SCORE {insertion_scoress_idgi_norm.mean(axis=1)}\n")
        file.write(str(compute_at))