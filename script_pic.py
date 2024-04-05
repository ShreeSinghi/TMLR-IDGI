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
import io
from torchvision.transforms import ToPILImage

import torch
from utils import load_preprocess, load_model, load_image_loader, compute_outputs
from metrics import insert_pixels, insertion_score, compute_pic_metric, ComputePicMetricError, aggregate_individual_pic_results

from multiprocessing import Pool

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

def estimate_image_entropy(image: np.ndarray) -> float:
    """Estimates the amount of information in a given image.

        Args:
            image: an image, which entropy should be estimated. The dimensions of the
                array should be [H, W, C] or [H, W] of type uint8.
        Returns:
            The estimated amount of information in the image.
    """
    buffer = io.BytesIO()
    pil_image = image
    pil_image.save(buffer, format='png', lossless=True)
    buffer.seek(0, os.SEEK_END)
    length = buffer.tell()
    buffer.close()
    return length

def load_images(dictionary, class_index):
    files = [os.path.join(opt.dataroot, r"val/images", file) for file in dictionary[class_index]]

    image_data = [preprocess(Image.open(path)) for path in files]
    image_data = np.array(image_data)

    return image_data

def compute_pic(saliencies):
    
    bokeh_images = insert_pixels(repeated_images,
                                 saliencies.view(-1, *saliencies.shape[2:]),
                                 repeated_blurs,
                                 linspace)
    num_processes = 16
    args = (model, preprocess(bokeh_images), indexs.repeat(N_DIVISIONS), batchSize)
    with Pool(num_processes) as p:

        entropies_promise = p.map_async(estimate_image_entropy, [ToPILImage()(image) for image in bokeh_images.view(-1, *bokeh_images.shape[2:])])

        pred_probas, pred_accs = compute_outputs(*args)

        entropies = np.array(entropies_promise.get()).reshape(len(repeated_images), N_DIVISIONS)
    print(entropies.shape)
    curve_y_compression_sics = []
    curve_y_compression_aics = []
    curve_y_msssim_sics = []
    curve_y_msssim_aics = []

    for i in tqdm(range(len(repeated_images))):
        try:
            curve_y_compression_sic = compute_pic_metric(repeated_images[i], pred_probas[i], bokeh_images[i], "compression", entropies=entropies[i])
            curve_y_compression_aic = compute_pic_metric(repeated_images[i], pred_accs[i], bokeh_images[i], "compression", entropies=entropies[i])

            curve_y_compression_sics.append(curve_y_compression_sic)
            curve_y_compression_aics.append(curve_y_compression_aic)
        except ComputePicMetricError:
            pass
        try:
            curve_y_msssim_sic = compute_pic_metric(repeated_images[i], pred_probas[i], bokeh_images[i], "msssim")
            curve_y_msssim_aic = compute_pic_metric(repeated_images[i], pred_accs[i], bokeh_images[i], "msssim")

            curve_y_msssim_sics.append(curve_y_msssim_sic)
            curve_y_msssim_aics.append(curve_y_msssim_aic)
        except ComputePicMetricError:
            pass

    del bokeh_images, args
    gc.collect()

    return (
        np.array(curve_y_compression_sics).reshape(len(compute_at), -1, len(curve_y_compression_sics[0])).astype(np.float16),
        np.array(curve_y_compression_aics).reshape(len(compute_at), -1, len(curve_y_compression_aics[0])).astype(np.float16),
        np.array(curve_y_msssim_sics).reshape(len(compute_at), -1, len(curve_y_msssim_sics[0])).astype(np.float16),
        np.array(curve_y_msssim_aics).reshape(len(compute_at), -1, len(curve_y_msssim_aics[0])).astype(np.float16)
    )

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

    with open(os.path.join(opt.dataroot, f"{opt.model}_probas.json"), 'r') as file:
        class_file_dict = json.load(file) # not really  a dict, its a list

    with open(os.path.join(opt.dataroot, f"{opt.model}_predictions.json"), 'r') as file:
        class_file_dict_all = json.load(file) # not really  a dict, its a list

    # class_file_dict has only those with over 80% confidence,  class_file_dict_all has all
    # we need to find those indices so we can select from produced saliencies
    
    if opt.method == "standard":
        compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
    if opt.method == "blurig":
        compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
    if opt.method == "gig":
        compute_at = [opt.n_steps]

    curve_y_compression_sicss_ig = []
    curve_y_compression_aicss_ig = []
    curve_y_msssim_sicss_ig = []
    curve_y_msssim_aicss_ig = []

    curve_y_compression_sicss_idgi = []
    curve_y_compression_aicss_idgi = []
    curve_y_msssim_sicss_idgi = []
    curve_y_msssim_aicss_idgi = []

    for class_idx in tqdm(range(opt.resume_epoch, 1000)):
        print("Processing class", class_idx)

        if len(class_file_dict[class_idx]) == 0:
            continue

        images = load([os.path.join(opt.dataroot, r"val/images", file ) for file in class_file_dict[class_idx]])

        blurs = load([os.path.join(opt.dataroot, r"val/blurred", file ) for file in class_file_dict[class_idx]])

        choose_indices = np.array([(x in class_file_dict[class_idx]) for x in class_file_dict_all[class_idx]], dtype=bool)

        preprocessed_images = preprocess(images)
        preprocessed_blurs = preprocess(blurs)

        saliencies_ig   = torch.from_numpy(np.stack([np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_ig",   f"class_{class_idx:03d}_steps_{n_step}.npy")) for n_step in compute_at]))[:, choose_indices]
        saliencies_idgi = torch.from_numpy(np.stack([np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_idgi", f"class_{class_idx:03d}_steps_{n_step}.npy")) for n_step in compute_at]))[:, choose_indices]

        indexs = torch.Tensor([class_idx] * (len(class_file_dict[class_idx]) * len(compute_at))).long()

        N_DIVISIONS = 25

        repeated_images = torch.from_numpy(images).repeat(len(compute_at), 1, 1, 1)
        repeated_blurs  = torch.from_numpy(blurs).repeat(len(compute_at), 1, 1, 1)

        linspace = np.linspace(0, images.shape[-1]*images.shape[-2], N_DIVISIONS).astype(int)
        
        curve_y_compression_sics_ig, curve_y_compression_aics_ig, curve_y_msssim_sics_ig, curve_y_msssim_aics_ig = compute_pic(saliencies_ig)
        curve_y_compression_sics_idgi, curve_y_compression_aics_idgi, curve_y_msssim_sics_idgi, curve_y_msssim_aics_idgi = compute_pic(saliencies_idgi)

        curve_y_compression_sicss_ig.append(curve_y_compression_sics_ig)
        curve_y_compression_aicss_ig.append(curve_y_compression_aics_ig)
        curve_y_msssim_sicss_ig.append(curve_y_msssim_sics_ig)
        curve_y_msssim_aicss_ig.append(curve_y_msssim_aics_ig)

        curve_y_compression_sicss_idgi.append(curve_y_compression_sics_idgi)
        curve_y_compression_aicss_idgi.append(curve_y_compression_aics_idgi)
        curve_y_msssim_sicss_idgi.append(curve_y_msssim_sics_idgi)
        curve_y_msssim_aicss_idgi.append(curve_y_msssim_aics_idgi)

        del saliencies_ig, saliencies_idgi, preprocessed_images, preprocessed_blurs, images, blurs
        gc.collect()
    curve_y_compression_sicss_ig = np.concatenate(curve_y_compression_sicss_ig, axis=1),
    curve_y_compression_aicss_ig = np.concatenate(curve_y_compression_aicss_ig, axis=1),
    curve_y_msssim_sicss_ig = np.concatenate(curve_y_msssim_sicss_ig, axis=1),
    curve_y_msssim_aicss_ig = np.concatenate(curve_y_msssim_aicss_ig, axis=1)

    curve_y_compression_sicss_idgi = np.concatenate(curve_y_compression_sicss_idgi, axis=1),
    curve_y_compression_aicss_idgi = np.concatenate(curve_y_compression_aicss_idgi, axis=1),
    curve_y_msssim_sicss_idgi = np.concatenate(curve_y_msssim_sicss_idgi, axis=1),
    curve_y_msssim_aicss_idgi = np.concatenate(curve_y_msssim_aicss_idgi, axis=1)
    print("yo3")

    os.mkdir(os.path.join(opt.dataroot, "results", opt.model, f"{opt.method}_ig"), exist_ok=True)
    os.mkdir(os.path.join(opt.dataroot, "results", opt.model, f"{opt.method}_idgi"), exist_ok=True)

    for j, n_step in enumerate(compute_at):
        compression_sic_ig = aggregate_individual_pic_results(curve_y_compression_sicss_ig[j], method="median")
        compression_aic_ig = aggregate_individual_pic_results(curve_y_compression_aicss_ig[j], method="mean")
        msssim_sic_ig = aggregate_individual_pic_results(curve_y_msssim_sicss_ig[j], method="median")
        msssim_aic_ig = aggregate_individual_pic_results(curve_y_msssim_aicss_ig[j], method="mean")

        compression_sic_idgi = aggregate_individual_pic_results(curve_y_compression_sicss_idgi[j], method="median")
        compression_aic_idgi = aggregate_individual_pic_results(curve_y_compression_aicss_idgi[j], method="mean")
        msssim_sic_idgi = aggregate_individual_pic_results(curve_y_msssim_sicss_idgi[j], method="median")
        msssim_aic_idgi = aggregate_individual_pic_results(curve_y_msssim_aicss_idgi[j], method="mean")

        print("compression_sic_ig", n_step, compression_sic_ig)
        print("compression_aic_ig", n_step, compression_aic_ig)
        print("msssim_sic_ig", n_step, msssim_sic_ig)
        print("msssim_aic_ig", n_step, msssim_aic_ig)

        print("compression_sic_idgi", n_step, compression_sic_idgi)
        print("compression_aic_idgi", n_step, compression_aic_idgi)
        print("msssim_sic_idgi", n_step, msssim_sic_idgi)
        print("msssim_aic_idgi", n_step, msssim_aic_idgi)