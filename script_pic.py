import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np
import json
import gc
from tqdm import tqdm
import io
from torchvision.transforms import ToPILImage

import torch
from utils import load_preprocess, load_model, load_image_loader, compute_outputs
from metrics import insert_pixels, compute_pic_metric, ComputePicMetricError, aggregate_individual_pic_results

from multiprocessing import Pool

cudnn.benchmark = True


def estimate_image_entropy(image: np.ndarray) -> float:
    buffer = io.BytesIO()
    pil_image = image
    pil_image.save(buffer, format='png', lossless=True)
    buffer.seek(0, os.SEEK_END)
    length = buffer.tell()
    buffer.close()
    return length

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

    curve_y_compression_sics = []
    curve_y_compression_aics = []
    curve_y_msssim_sics = []
    curve_y_msssim_aics = []

    for i in range(len(repeated_images)):
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

    rets = curve_y_compression_sics, curve_y_compression_aics, curve_y_msssim_sics, curve_y_msssim_aics
    rets = [(np.array(ret).reshape(len(compute_at), -1, len(ret[0])).astype(np.float16) if len(ret)>0 else None) for ret in rets]
    return rets

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
    print(f"Computing PIC for {opt.model} using {opt.method} method")

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
        
        if curve_y_compression_sics_ig is not None:
            curve_y_compression_sicss_ig.append(curve_y_compression_sics_ig)
        if curve_y_compression_aics_ig is not None:
            curve_y_compression_aicss_ig.append(curve_y_compression_aics_ig)
        if curve_y_msssim_sics_ig is not None:
            curve_y_msssim_sicss_ig.append(curve_y_msssim_sics_ig)
        if curve_y_msssim_aics_ig is not None:
            curve_y_msssim_aicss_ig.append(curve_y_msssim_aics_ig)

        if curve_y_compression_sics_idgi is not None:
            curve_y_compression_sicss_idgi.append(curve_y_compression_sics_idgi)
        if curve_y_compression_aics_idgi is not None:
            curve_y_compression_aicss_idgi.append(curve_y_compression_aics_idgi)
        if curve_y_msssim_sics_idgi is not None:
            curve_y_msssim_sicss_idgi.append(curve_y_msssim_sics_idgi)
        if curve_y_msssim_aics_idgi is not None:
            curve_y_msssim_aicss_idgi.append(curve_y_msssim_aics_idgi)

        del saliencies_ig, saliencies_idgi, preprocessed_images, preprocessed_blurs, images, blurs
        gc.collect()

        if (class_idx+1) % 50 == 0:
            curve_y_compression_sicss_ig_ = np.concatenate(curve_y_compression_sicss_ig, axis=1)
            curve_y_compression_aicss_ig_ = np.concatenate(curve_y_compression_aicss_ig, axis=1)
            curve_y_msssim_sicss_ig_ = np.concatenate(curve_y_msssim_sicss_ig, axis=1)
            curve_y_msssim_aicss_ig_ = np.concatenate(curve_y_msssim_aicss_ig, axis=1)

            curve_y_compression_sicss_idgi_ = np.concatenate(curve_y_compression_sicss_idgi, axis=1)
            curve_y_compression_aicss_idgi_ = np.concatenate(curve_y_compression_aicss_idgi, axis=1)
            curve_y_msssim_sicss_idgi_ = np.concatenate(curve_y_msssim_sicss_idgi, axis=1)
            curve_y_msssim_aicss_idgi_ = np.concatenate(curve_y_msssim_aicss_idgi, axis=1)

            for j, n_step in enumerate(compute_at):
                compression_sic_ig = aggregate_individual_pic_results(curve_y_compression_sicss_ig_[j], method="median")
                compression_aic_ig = aggregate_individual_pic_results(curve_y_compression_aicss_ig_[j], method="mean")
                msssim_sic_ig = aggregate_individual_pic_results(curve_y_msssim_sicss_ig_[j], method="median")
                msssim_aic_ig = aggregate_individual_pic_results(curve_y_msssim_aicss_ig_[j], method="mean")

                compression_sic_idgi = aggregate_individual_pic_results(curve_y_compression_sicss_idgi_[j], method="median")
                compression_aic_idgi = aggregate_individual_pic_results(curve_y_compression_aicss_idgi_[j], method="mean")
                msssim_sic_idgi = aggregate_individual_pic_results(curve_y_msssim_sicss_idgi_[j], method="median")
                msssim_aic_idgi = aggregate_individual_pic_results(curve_y_msssim_aicss_idgi_[j], method="mean")

                os.makedirs("results", exist_ok=True)
                with open(f"results/{opt.model}_{opt.method}_ig_{n_step}.txt", 'w') as file:
                    file.write(f"compression_sic_ig {compression_sic_ig}\n")
                    file.write(f"compression_aic_ig {compression_aic_ig}\n")
                    file.write(f"msssim_sic_ig {msssim_sic_ig}\n")
                    file.write(f"msssim_aic_ig {msssim_aic_ig}\n")
                    file.write(f"step {class_idx}\n")

                with open(f"results/{opt.model}_{opt.method}_idgi_{n_step}.txt", 'w') as file:
                    file.write(f"compression_sic_idgi {compression_sic_idgi}\n")
                    file.write(f"compression_aic_idgi {compression_aic_idgi}\n")
                    file.write(f"msssim_sic_idgi {msssim_sic_idgi}\n")
                    file.write(f"msssim_aic_idgi {msssim_aic_idgi}\n")
                    file.write(f"step {class_idx}\n")