import numpy as np
from PIL import Image
import os
import json
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm


f = lambda a, x: np.log2(1+a*x)
g = lambda a, x: (np.exp2(x)-1)/a

def compress(arr):
    arr = f(64, f(64, f(64, arr)))
    maxi = arr.max((1,2), keepdims=True)
    arr = (255*arr/maxi).astype(np.uint8)
    return arr, maxi

def decompress(arr, maxi):
    arr = arr.astype(np.float64)*maxi/255
    arr = g(64, g(64, g(64, arr)))
    return arr

def save(arr, maxi, folder):
    os.makedirs(folder, exist_ok=True)
    for i, image in enumerate(arr):
        img = Image.fromarray(image)
        img.save(os.path.join(folder, f"{i}.png"))
    np.save(os.path.join(folder, "maxi"), maxi)

def load(folder):
    arr = [Image.open(os.path.join(folder, name)) for name in os.listdir(folder) if name.endswith(".png")]
    arr = sorted(arr, key=lambda x: int(x.filename.split("/")[-1].split(".")[0]))
    arr = np.array([np.array(image) for image in arr])
    maxi = np.load(os.path.join(folder, "maxi.npy"))
    arr = decompress(arr, maxi)
    return arr

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')

    opt = parser.parse_args()

    subfolders = [os.path.join(opt.dataroot, name) for name in os.listdir(opt.dataroot)]
    files = [os.path.join(subfolder, name) for subfolder in subfolders for name in os.listdir(subfolder) if name.endswith(".npy")]

    for file in tqdm(files):
        arr = np.load(file).astype(np.float64)
        arr /= arr.sum((1,2), keepdims=True)
        arr_compress, maxi = compress(arr)
        save(arr_compress, maxi, file[:-4])